import argparse, glob, hashlib, itertools, json, math, os, random, sys, tempfile, time, uuid
from datetime import datetime, timezone

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util


CHECKPOINT_SCHEMA_VERSION = 3
HEAD_SCHEMA_VERSION = 3
PAYLOAD_SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "1.33"

CHECKPOINT_FORMAT = "tmt-galia-checkpoint"
HEAD_FORMAT = "tmt-galia-head"
RECOVERY_POLICY = "current_then_previous"
CHECKPOINT_FAILPOINT_ENV = "TMT_CHECKPOINT_FAILPOINT"
CONFIG_BINDING = "full-v1.22"
LEGACY_LOAD_POLICY = "explicit-opt-in-v1.27"

def _migrate_checkpoint_manifest_v1_to_v2(doc: dict) -> dict:
    migrated = dict(doc)
    migrated["schema_version"] = 2
    migrated.setdefault("parent_generation", None)
    migrated.setdefault("checkpoint_kind", "full")
    return migrated

def _migrate_checkpoint_manifest_v2_to_v3(doc: dict) -> dict:
    migrated = dict(doc)
    migrated["schema_version"] = 3
    migrated.setdefault("payload_schema_version", 1)
    migrated.setdefault("protocol_version", "legacy")
    migrated.setdefault("commit_id", None)
    migrated.setdefault("commit_seq", 0)
    migrated.setdefault("model_config", None)
    migrated.setdefault("config_binding", "unbound-legacy")
    return migrated

def _migrate_head_v1_to_v2(doc: dict) -> dict:
    migrated = dict(doc)
    migrated["schema_version"] = 2
    migrated.setdefault("recovery_policy", RECOVERY_POLICY)
    migrated.setdefault(
        "commit_id",
        f"legacy-{migrated.get('current_generation', 'unknown')}",
    )
    return migrated

def _migrate_head_v2_to_v3(doc: dict) -> dict:
    migrated = dict(doc)
    migrated["schema_version"] = 3
    migrated.setdefault("protocol_version", "legacy")
    migrated.setdefault("commit_seq", 0)
    migrated.setdefault("recovery_policy", RECOVERY_POLICY)
    return migrated

CHECKPOINT_MANIFEST_MIGRATIONS = {
    1: _migrate_checkpoint_manifest_v1_to_v2,
    2: _migrate_checkpoint_manifest_v2_to_v3,
}
HEAD_MIGRATIONS = {
    1: _migrate_head_v1_to_v2,
    2: _migrate_head_v2_to_v3,
}

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def __call__(self, x: mx.array): return self.embed(x)

class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def __call__(self, x: mx.array): return self.decode(x), mx.sigmoid(self.stop(x))

class Layer(nn.Module):
    def __init__(self, dim: int, spread: int):
        super().__init__()

        halflives = mx.exp(mx.linspace(0.0, math.log(float(spread)), dim))
        retention = mx.exp(-math.log(2.0) / halflives)
        self.decay = mx.log(retention) - mx.log1p(-retention)
    
        self.states = mx.zeros((dim, ))
        self.decaytrace = mx.zeros((dim, ))
        self.embedtrace = mx.zeros((256, dim))
        
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias = False)
        self.silu = nn.SiLU()

        self.freeze(keys = ['states', 'decaytrace', 'embedtrace'], recurse = False)        

    def __call__(self, enc: mx.array, x: mx.array, dummy: mx.array):
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + enc + dummy

        return x + self.silu(self.weights(self.norm(state))), state, decay

class Model(nn.Module):
    def __init__(self, dim: int, layers: int, spread: int, temp: float, lr: float, lrbegin: int, lrend: int):
        super().__init__()
        self.dim = dim
        self.layercount = layers
        self.temp = temp
        self.spread = spread
        self.lr = lr
        self.lrbegin = lrbegin
        self.lrend = lrend

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.layers = [Layer(dim, spread) for _ in range(layers)]

        def lrfn(step: mx.array):
            progress = mx.clip(
                (step.astype(mx.float32) + 1.0 - float(lrbegin)) / float(lrend - lrbegin),
                0.0, 1.0
            )
            return lr * (1.0 - 0.9 * progress)

        self.optimizer = opt.AdamW(learning_rate = lrfn)
        self._last_checkpoint_event = None
        self._legacy_unbound_taint = False
        self._legacy_unbound_source = None

    def sample(self, output: mx.array):
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(256.0))

        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy)).item()
        return mx.random.categorical(output / temp)

    def evaluate(self): mx.eval(*[layer.states for layer in self.layers])

    def reset(self):
        for layer in self.layers:
            layer.states = mx.zeros((self.dim, ))

            layer.decaytrace = mx.zeros((self.dim, ))
            layer.embedtrace = mx.zeros((256, self.dim))

        self.evaluate()

    def step(self, c: mx.array, dummies: mx.array | None = None, frozen: bool = False):
        if dummies is None: dummies = [mx.zeros((self.dim, )) for _ in range(self.layercount)]

        enc = self.encoder(c)
        x = enc
            
        states, decays = [], []

        for i, layer in enumerate(self.layers):
            x, state, decay = layer(enc, x, dummies[i])
            if frozen: layer.states = mx.stop_gradient(state)

            states.append(state)
            decays.append(decay)

        return (x, states, decays), self.decoder(x)

    def __call__(self, currb: int, nextb: int | None, end: bool, frozen: bool):
        c = mx.array(currb)

        if frozen:
            _, (output, stop) = self.step(c, frozen = True)

            self.evaluate()
            return self.sample(output).item(), stop.item()

        p = self.trainable_parameters()

        def fwd(params, dummies: list[mx.array]):
            self.update(params)

            (x, states, decays), (output, stop) = self.step(c, dummies)

            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4))
            if nextb is not None:
                n = mx.array(nextb)
                tgt = mx.stop_gradient(self.encoder(n))

                loss = loss + mx.mean(mx.square(x - tgt))
                loss = loss - output[n] + mx.logsumexp(output)

                loss = loss + mx.mean(mx.square(stop - mx.array([1.0 if end else 0.0])))
                
            # loss = variance loss + pred mse loss + crossentropy loss + stop mse loss
            return loss, (states, decays, output, stop)

        (_, (states, decays, output, stop)), (grads, dlds_s) = mx.value_and_grad(
            fwd, argnums = (0, 1)
        )(p, [mx.zeros((self.dim, )) for _ in range(self.layercount)])

        self.update(p)

        for i, layer in enumerate(self.layers):
            dlds = dlds_s[i]

            embedtrace = (layer.embedtrace * decays[i]) + (mx.arange(256) == c)[:, None].astype(mx.float32)
            grads["encoder"]["embed"]["weight"] += dlds * (layer.embedtrace * decays[i])
            
            decaytrace = (decays[i] * layer.decaytrace) + (decays[i] * (1.0 - decays[i]) * layer.states)
            grads["layers"][i]["decay"] = dlds * decaytrace

            layer.states = mx.stop_gradient(states[i])

            layer.decaytrace = mx.stop_gradient(decaytrace)
            layer.embedtrace = mx.stop_gradient(embedtrace)
            
            mx.eval(layer.states, layer.decaytrace, layer.embedtrace)

        self.optimizer.update(self, grads)
        mx.eval(self.parameters(), self.optimizer.state)

        return self.sample(output).item(), stop.item()

    def _mark_legacy_unbound_taint(self, source: str):
        self._legacy_unbound_taint = True
        self._legacy_unbound_source = source

    def _clear_legacy_unbound_taint(self):
        self._legacy_unbound_taint = False
        self._legacy_unbound_source = None

    @property
    def last_checkpoint_event(self):
        """Last checkpoint event metadata; intentionally excluded from MLX parameters."""
        return self._last_checkpoint_event

    def checkpoint_authority_status(self) -> dict:
        return {
            "legacy_unbound_taint": bool(self._legacy_unbound_taint),
            "legacy_unbound_source": self._legacy_unbound_source,
            "authoritative_save_allowed": not self._legacy_unbound_taint,
        }

    def _checkpoint_model_config(self) -> dict:
        return {
            "dim": int(self.dim),
            "layers": int(self.layercount),
            "spread": int(self.spread),
            "temp": float(self.temp),
            "lr": float(self.lr),
            "lrbegin": int(self.lrbegin),
            "lrend": int(self.lrend),
        }

    def _checkpoint_parts(self, data: dict):
        model, opts = {}, {}
        states, decaytraces, embedtraces = {}, {}, {}

        for k, v in data.items():
            if k.startswith("m."):
                model[k[2:]] = v
            elif k.startswith("o."):
                opts[k[2:]] = v
            elif k.startswith("state."):
                states[int(k.split(".")[1])] = v
            elif k.startswith("decaytrace."):
                decaytraces[int(k.split(".")[1])] = v
            elif k.startswith("embedtrace."):
                embedtraces[int(k.split(".")[1])] = v
            else:
                raise ValueError(f"Unknown checkpoint key: {k!r}")

        expected_model = dict(util.tree_flatten(self.parameters()))
        if set(model) != set(expected_model):
            missing = sorted(set(expected_model) - set(model))
            extra = sorted(set(model) - set(expected_model))
            raise ValueError(
                f"Checkpoint model schema mismatch; missing={missing}, extra={extra}"
            )

        for k, current in expected_model.items():
            if tuple(model[k].shape) != tuple(current.shape):
                raise ValueError(
                    f"Checkpoint model shape mismatch for {k!r}: "
                    f"{tuple(model[k].shape)} != {tuple(current.shape)}"
                )

        expected_layers = set(range(self.layercount))
        for label, values in (
            ("state", states),
            ("decaytrace", decaytraces),
            ("embedtrace", embedtraces),
        ):
            if set(values) != expected_layers:
                missing = sorted(expected_layers - set(values))
                extra = sorted(set(values) - expected_layers)
                raise ValueError(
                    f"Checkpoint {label} schema mismatch; "
                    f"missing={missing}, extra={extra}"
                )

        for i, layer in enumerate(self.layers):
            if tuple(states[i].shape) != tuple(layer.states.shape):
                raise ValueError(f"Checkpoint state shape mismatch for layer {i}")
            if tuple(decaytraces[i].shape) != tuple(layer.decaytrace.shape):
                raise ValueError(f"Checkpoint decaytrace shape mismatch for layer {i}")
            if tuple(embedtraces[i].shape) != tuple(layer.embedtrace.shape):
                raise ValueError(f"Checkpoint embedtrace shape mismatch for layer {i}")

        try:
            model_tree = util.tree_unflatten(list(model.items()))
            opts_tree = util.tree_unflatten(list(opts.items())) if opts else None
        except Exception as exc:
            raise ValueError("Checkpoint tree structure is invalid") from exc

        return model_tree, opts_tree, states, decaytraces, embedtraces

    @staticmethod
    def _fsync_directory(path: str):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _json_load(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {path!r}")
        return value

    def _json_save_atomic(self, path: str, value: dict):
        parent = os.path.dirname(path) or os.curdir
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, sort_keys=True, separators=(",", ":"))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            self._fsync_directory(parent)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @staticmethod
    def _head_path(target: str) -> str:
        return f"{target}.head.json"

    @staticmethod
    def _audit_path(target: str) -> str:
        return f"{target}.audit.jsonl"

    @staticmethod
    def _generation_paths(target: str, generation_id: str) -> tuple[str, str]:
        stem = f"{target}.g-{generation_id}"
        return f"{stem}.safetensors", f"{stem}.manifest.json"

    @staticmethod
    def _migrate_document(
        doc: dict,
        *,
        kind: str,
        current_version: int,
        migrations: dict,
    ) -> tuple[dict, int]:
        version = doc.get("schema_version")
        if not isinstance(version, int) or version < 1:
            raise ValueError(f"{kind} schema_version is invalid: {version!r}")
        if version > current_version:
            raise ValueError(f"Unsupported future {kind} schema version: {version!r}")
        original_version = version
        migrated = dict(doc)
        while version < current_version:
            migrate = migrations.get(version)
            if migrate is None:
                raise ValueError(
                    f"No explicit {kind} migration registered for "
                    f"schema {version} -> {version + 1}"
                )
            migrated = migrate(migrated)
            new_version = migrated.get("schema_version")
            if new_version != version + 1:
                raise ValueError(
                    f"{kind} migration {version} did not advance exactly one version"
                )
            version = new_version
        return migrated, original_version

    def _append_audit(self, target: str, event: str, **fields) -> bool:
        record = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        record.update(fields)
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        path = self._audit_path(target)
        parent = os.path.dirname(path) or os.curdir
        os.makedirs(parent, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("audit append made no progress")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            return True
        except OSError as exc:
            print(f"checkpoint audit append failed: {exc}", file=sys.stderr)
            return False

    @staticmethod
    def _failpoint(stage: str):
        if os.environ.get("TMT_CHECKPOINT_FAILPOINT") == stage:
            raise RuntimeError(f"Simulated checkpoint failpoint: {stage}")

    def _read_head(self, target: str) -> dict:
        raw = self._json_load(self._head_path(target))
        if raw.get("format") != HEAD_FORMAT:
            raise ValueError("Unknown checkpoint HEAD format")
        head, original_schema = self._migrate_document(
            raw, kind="HEAD", current_version=HEAD_SCHEMA_VERSION,
            migrations=HEAD_MIGRATIONS,
        )
        if original_schema != HEAD_SCHEMA_VERSION:
            self._append_audit(
                target, "head_schema_migrated",
                from_schema=original_schema, to_schema=HEAD_SCHEMA_VERSION,
            )
        if head.get("recovery_policy") != RECOVERY_POLICY:
            raise ValueError("Unsupported checkpoint recovery policy")
        current = head.get("current_generation")
        previous = head.get("previous_generation")
        commit_id = head.get("commit_id")
        commit_seq = head.get("commit_seq")
        protocol_version = head.get("protocol_version")
        if not isinstance(current, str) or not current:
            raise ValueError("HEAD is missing current_generation")
        if previous is not None and (not isinstance(previous, str) or not previous):
            raise ValueError("HEAD previous_generation is invalid")
        if previous is not None and previous == current:
            raise ValueError("HEAD current_generation and previous_generation must differ")
        if not isinstance(commit_id, str) or not commit_id:
            raise ValueError("HEAD commit_id is invalid")
        if (
            not isinstance(commit_seq, int)
            or isinstance(commit_seq, bool)
            or commit_seq < 0
        ):
            raise ValueError("HEAD commit_seq is invalid")
        if original_schema == HEAD_SCHEMA_VERSION and commit_seq < 1:
            raise ValueError("Native schema-3 HEAD commit_seq must be >= 1")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise ValueError("HEAD protocol_version is invalid")
        return head

    def _read_generation_manifest(
        self,
        target: str,
        generation_id: str,
        expected_commit_id: str | None = None,
        expected_commit_seq: int | None = None,
    ):
        checkpoint_path, manifest_path = self._generation_paths(target, generation_id)
        raw = self._json_load(manifest_path)
        if raw.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Unknown checkpoint manifest format")
        manifest, original_schema = self._migrate_document(
            raw, kind="checkpoint manifest",
            current_version=CHECKPOINT_SCHEMA_VERSION,
            migrations=CHECKPOINT_MANIFEST_MIGRATIONS,
        )
        if original_schema != CHECKPOINT_SCHEMA_VERSION:
            self._append_audit(
                target, "manifest_schema_migrated",
                generation_id=generation_id,
                from_schema=original_schema,
                to_schema=CHECKPOINT_SCHEMA_VERSION,
            )
        if manifest.get("payload_schema_version") != PAYLOAD_SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint payload schema version")
        if manifest.get("generation_id") != generation_id:
            raise ValueError("Checkpoint generation ID mismatch")
        if manifest.get("checkpoint_file") != os.path.basename(checkpoint_path):
            raise ValueError("Checkpoint filename does not match manifest")
        if manifest.get("checkpoint_kind") != "full":
            raise ValueError("Unsupported checkpoint kind")
        if expected_commit_id is not None and manifest.get("commit_id") != expected_commit_id:
            raise ValueError("Checkpoint commit_id does not match HEAD")
        if expected_commit_seq is not None and manifest.get("commit_seq") != expected_commit_seq:
            raise ValueError("Checkpoint commit_seq does not match HEAD")
        if manifest.get("config_binding") != CONFIG_BINDING:
            raise ValueError("Checkpoint lacks v1.22 full constructor/schedule binding")
        if manifest.get("model_config") != self._checkpoint_model_config():
            raise ValueError("Checkpoint model/runtime configuration mismatch")
        protocol_version = manifest.get("protocol_version")
        commit_id = manifest.get("commit_id")
        commit_seq = manifest.get("commit_seq")
        parent_generation = manifest.get("parent_generation")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise ValueError("Checkpoint protocol_version is invalid")
        if not isinstance(commit_id, str) or not commit_id:
            raise ValueError("Checkpoint commit_id is invalid")
        if not isinstance(commit_seq, int) or isinstance(commit_seq, bool) or commit_seq < 1:
            raise ValueError("Checkpoint commit_seq is invalid")
        if parent_generation is not None and (
            not isinstance(parent_generation, str) or not parent_generation
        ):
            raise ValueError("Checkpoint parent_generation is invalid")
        return manifest, checkpoint_path

    def _validate_fallback_lineage(self, target: str, head: dict):
        current_generation = head.get("current_generation")
        previous_generation = head.get("previous_generation")
        if not previous_generation:
            raise ValueError("HEAD has no previous_generation for fallback")
        manifest, _ = self._read_generation_manifest(
            target,
            current_generation,
            expected_commit_id=head.get("commit_id"),
            expected_commit_seq=head.get("commit_seq"),
        )
        parent_generation = manifest.get("parent_generation")
        if parent_generation != previous_generation:
            raise ValueError(
                "Fallback lineage mismatch: HEAD previous_generation "
                f"{previous_generation!r} != current manifest parent_generation "
                f"{parent_generation!r}"
            )
        return manifest

    def _validate_generation(
        self,
        target: str,
        generation_id: str,
        expected_commit_id: str | None = None,
        expected_commit_seq: int | None = None,
    ):
        manifest, checkpoint_path = self._read_generation_manifest(
            target,
            generation_id,
            expected_commit_id=expected_commit_id,
            expected_commit_seq=expected_commit_seq,
        )
        expected_size = manifest.get("size_bytes")
        actual_size = os.path.getsize(checkpoint_path)
        if expected_size != actual_size:
            raise ValueError(f"Checkpoint size mismatch: {actual_size} != {expected_size}")
        expected_hash = manifest.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("Checkpoint manifest SHA-256 is invalid")
        if self._sha256_file(checkpoint_path) != expected_hash:
            raise ValueError("Checkpoint SHA-256 mismatch")
        data = mx.load(checkpoint_path)
        parts = self._checkpoint_parts(data)
        return parts, manifest

    def _snapshot_live_checkpoint_state(self):
        return (
            self.parameters(),
            self.optimizer.state,
            [layer.states for layer in self.layers],
            [layer.decaytrace for layer in self.layers],
            [layer.embedtrace for layer in self.layers],
        )

    def _restore_live_checkpoint_state(self, snapshot):
        model_tree, optimizer_state, states, decaytraces, embedtraces = snapshot
        self.update(model_tree)
        self.optimizer.state = optimizer_state
        for i, layer in enumerate(self.layers):
            layer.states = states[i]
            layer.decaytrace = decaytraces[i]
            layer.embedtrace = embedtraces[i]
        mx.eval(
            self.parameters(), self.optimizer.state,
            *[layer.states for layer in self.layers],
            *[layer.decaytrace for layer in self.layers],
            *[layer.embedtrace for layer in self.layers],
        )

    def _apply_checkpoint(self, parts):
        model_tree, opts_tree, states, decaytraces, embedtraces = parts
        snapshot = self._snapshot_live_checkpoint_state()
        try:
            self.update(model_tree)
            self._failpoint("apply_after_model")
            if opts_tree is not None:
                self.optimizer.state = opts_tree
            self._failpoint("apply_after_optimizer")
            for i, layer in enumerate(self.layers):
                layer.states = states[i]
                layer.decaytrace = decaytraces[i]
                layer.embedtrace = embedtraces[i]
            self._failpoint("apply_after_recurrent")
            self._failpoint("apply_before_eval")
            mx.eval(
                self.parameters(), self.optimizer.state,
                *[layer.states for layer in self.layers],
                *[layer.decaytrace for layer in self.layers],
                *[layer.embedtrace for layer in self.layers],
            )
        except Exception as apply_exc:
            try:
                self._restore_live_checkpoint_state(snapshot)
            except Exception as rollback_exc:
                raise RuntimeError(
                    "Checkpoint apply failed and in-memory rollback also failed"
                ) from rollback_exc
            raise apply_exc

    def _current_valid_generation(self, target: str) -> tuple[str | None, dict | None]:
        if not os.path.exists(self._head_path(target)):
            return None, None
        head = self._read_head(target)
        candidates = [
            ("current", head["current_generation"]),
            ("previous", head.get("previous_generation")),
        ]
        errors = []
        for role, generation_id in candidates:
            if not generation_id:
                continue
            try:
                if role == "current":
                    self._validate_generation(
                        target, generation_id,
                        expected_commit_id=head.get("commit_id"),
                        expected_commit_seq=head.get("commit_seq"),
                    )
                else:
                    self._validate_fallback_lineage(target, head)
                    self._validate_generation(target, generation_id)
                if role == "previous":
                    self._append_audit(
                        target, "save_lineage_fallback",
                        failed_generation=head["current_generation"],
                        recovered_generation=generation_id,
                    )
                return generation_id, head
            except Exception as exc:
                errors.append(f"{generation_id}: {exc}")
                self._append_audit(
                    target, "generation_validation_failed",
                    operation="save", role=role, generation_id=generation_id,
                    error_type=type(exc).__name__, error=str(exc),
                )
        self._append_audit(target, "save_refused_corrupt_lineage", errors=errors)
        raise ValueError(
            "Existing checkpoint lineage has no valid generation; refusing save. "
            + " | ".join(errors)
        )

    def _prune_generations(self, target: str, keep: set[str]):
        prefix = f"{target}.g-"
        for checkpoint_path in glob.glob(f"{prefix}*.safetensors"):
            name = checkpoint_path[len(prefix):]
            if not name.endswith(".safetensors"):
                continue
            generation_id = name[:-len(".safetensors")]
            if generation_id in keep:
                continue
            manifest_path = f"{prefix}{generation_id}.manifest.json"
            for candidate_path in (checkpoint_path, manifest_path):
                try:
                    if os.path.exists(candidate_path):
                        os.unlink(candidate_path)
                except OSError:
                    pass

    def _next_commit_seq(self, target: str) -> int:
        if not os.path.exists(self._head_path(target)):
            return 1
        return self._read_head(target)["commit_seq"] + 1

    def checkpoint_status(self, path: str) -> dict:
        target = os.path.abspath(os.path.expanduser(path))
        result = {
            "logical_path": target,
            "head_present": os.path.exists(self._head_path(target)),
            "legacy_present": os.path.exists(target),
            "loadable": False,
            "selected_generation": None,
            "fallback_required": False,
            "generations": [],
        }
        if not result["head_present"]:
            result["mode"] = "legacy-unbound" if result["legacy_present"] else "empty"
            result["loadable"] = False
            if result["legacy_present"]:
                result["legacy_unbound_present"] = True
                result["config_binding"] = "legacy-unbound"
                result["legacy_load_policy"] = LEGACY_LOAD_POLICY
                result["requires_explicit_legacy_authorization"] = True
                result["loadable_with_explicit_legacy_authorization"] = True
            return result
        result["mode"] = "generation"
        try:
            head = self._read_head(target)
        except Exception as exc:
            result["head_error"] = str(exc)
            return result
        result["head"] = {
            "schema_version": head.get("schema_version"),
            "protocol_version": head.get("protocol_version"),
            "commit_id": head.get("commit_id"),
            "commit_seq": head.get("commit_seq"),
            "current_generation": head.get("current_generation"),
            "previous_generation": head.get("previous_generation"),
        }
        for role, generation_id in (
            ("current", head.get("current_generation")),
            ("previous", head.get("previous_generation")),
        ):
            if not generation_id:
                continue
            entry = {"role": role, "generation_id": generation_id}
            try:
                if role == "current":
                    _, manifest = self._validate_generation(
                        target, generation_id,
                        expected_commit_id=head.get("commit_id"),
                        expected_commit_seq=head.get("commit_seq"),
                    )
                else:
                    self._validate_fallback_lineage(target, head)
                    _, manifest = self._validate_generation(target, generation_id)
                entry["valid"] = True
                entry["schema_version"] = manifest.get("schema_version")
                entry["commit_id"] = manifest.get("commit_id")
                entry["commit_seq"] = manifest.get("commit_seq")
                entry["model_config"] = manifest.get("model_config")
                if result["selected_generation"] is None:
                    result["selected_generation"] = generation_id
                    result["loadable"] = True
                    result["fallback_required"] = role == "previous"
            except Exception as exc:
                entry["valid"] = False
                entry["error"] = str(exc)
            result["generations"].append(entry)
        return result

    def save(self, path: str):
        target = os.path.abspath(os.path.expanduser(path))
        if self._legacy_unbound_taint:
            self._append_audit(
                target, "save_refused_legacy_unbound_taint",
                legacy_unbound_source=self._legacy_unbound_source,
            )
            raise ValueError(
                "Refusing authoritative save from a legacy-unbound checkpoint. "
                "The loaded state has unknown original constructor/LR schedule. "
                "Load an authoritative generation before saving."
            )

        data = {}
        for k, v in util.tree_flatten(self.parameters()):
            data[f"m.{k}"] = v
        for k, v in util.tree_flatten(self.optimizer.state):
            data[f"o.{k}"] = v
        for i, layer in enumerate(self.layers):
            data[f"state.{i}"] = layer.states
            data[f"decaytrace.{i}"] = layer.decaytrace
            data[f"embedtrace.{i}"] = layer.embedtrace

        parent = os.path.dirname(target) or os.curdir
        os.makedirs(parent, exist_ok=True)

        previous_generation, old_head = self._current_valid_generation(target)
        commit_seq = self._next_commit_seq(target)
        commit_id = uuid.uuid4().hex
        generation_id = uuid.uuid4().hex
        checkpoint_path, manifest_path = self._generation_paths(target, generation_id)

        fd, tmp = tempfile.mkstemp(
            prefix=f".{os.path.basename(checkpoint_path)}.",
            suffix=".safetensors", dir=parent,
        )
        os.close(fd)
        try:
            mx.save_safetensors(tmp, data)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            staged = mx.load(tmp)
            self._checkpoint_parts(staged)
            digest = self._sha256_file(tmp)
            size_bytes = os.path.getsize(tmp)
            os.replace(tmp, checkpoint_path)
            self._fsync_directory(parent)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        self._failpoint("after_generation_promote")

        manifest = {
            "format": CHECKPOINT_FORMAT,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "generation_id": generation_id,
            "parent_generation": previous_generation,
            "checkpoint_kind": "full",
            "commit_id": commit_id,
            "commit_seq": commit_seq,
            "checkpoint_file": os.path.basename(checkpoint_path),
            "sha256": digest,
            "size_bytes": size_bytes,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_binding": CONFIG_BINDING,
            "model_config": self._checkpoint_model_config(),
        }
        self._json_save_atomic(manifest_path, manifest)
        self._failpoint("after_manifest_promote")

        self._validate_generation(
            target, generation_id,
            expected_commit_id=commit_id, expected_commit_seq=commit_seq,
        )

        head = {
            "format": HEAD_FORMAT,
            "schema_version": HEAD_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "commit_id": commit_id,
            "commit_seq": commit_seq,
            "recovery_policy": RECOVERY_POLICY,
            "current_generation": generation_id,
            "previous_generation": previous_generation,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        self._failpoint("before_head_commit")
        self._json_save_atomic(self._head_path(target), head)

        self._last_checkpoint_event = {
            "operation": "save",
            "status": "committed",
            "generation_id": generation_id,
            "previous_generation": previous_generation,
            "commit_id": commit_id,
            "commit_seq": commit_seq,
        }
        self._append_audit(
            target, "save_commit",
            commit_id=commit_id, commit_seq=commit_seq,
            current_generation=generation_id,
            previous_generation=previous_generation,
        )
        self._failpoint("after_head_commit")

        keep = {generation_id}
        if previous_generation is not None:
            keep.add(previous_generation)
        if old_head is not None:
            keep.add(old_head["current_generation"])
            if old_head.get("previous_generation"):
                keep.add(old_head["previous_generation"])
        self._prune_generations(target, keep)

    def load(self, path: str, *, allow_legacy_unbound: bool = False):
        target = os.path.abspath(os.path.expanduser(path))
        head_path = self._head_path(target)

        if os.path.exists(head_path):
            head = self._read_head(target)
            candidates = [
                ("current", head["current_generation"]),
                ("previous", head.get("previous_generation")),
            ]
            errors = []
            seen = set()
            for role, generation_id in candidates:
                if not generation_id or generation_id in seen:
                    continue
                seen.add(generation_id)
                try:
                    if role == "current":
                        parts, manifest = self._validate_generation(
                            target, generation_id,
                            expected_commit_id=head.get("commit_id"),
                            expected_commit_seq=head.get("commit_seq"),
                        )
                    else:
                        self._validate_fallback_lineage(target, head)
                        parts, manifest = self._validate_generation(target, generation_id)
                except Exception as exc:
                    errors.append(f"{generation_id}: validation: {exc}")
                    self._append_audit(
                        target, "generation_validation_failed",
                        operation="load", role=role, generation_id=generation_id,
                        error_type=type(exc).__name__, error=str(exc),
                    )
                    continue

                try:
                    self._apply_checkpoint(parts)
                except Exception as exc:
                    errors.append(f"{generation_id}: apply: {exc}")
                    self._append_audit(
                        target, "generation_apply_failed_rolled_back",
                        operation="load", role=role, generation_id=generation_id,
                        error_type=type(exc).__name__, error=str(exc),
                    )
                    continue

                self._clear_legacy_unbound_taint()
                self._last_checkpoint_event = {
                    "operation": "load",
                    "status": "fallback" if role == "previous" else "current",
                    "generation_id": generation_id,
                    "fallback_used": role == "previous",
                    "head_commit_id": head.get("commit_id"),
                    "head_commit_seq": head.get("commit_seq"),
                    "manifest_schema_version": manifest.get("schema_version"),
                    "payload_schema_version": manifest.get("payload_schema_version"),
                    "model_config": manifest.get("model_config"),
                    "errors": list(errors),
                }
                if role == "previous":
                    self._append_audit(
                        target, "load_fallback",
                        commit_id=head["commit_id"],
                        failed_generation=head["current_generation"],
                        recovered_generation=generation_id,
                    )
                else:
                    self._append_audit(
                        target, "load_current",
                        commit_id=head["commit_id"], generation_id=generation_id,
                    )
                return

            self._append_audit(
                target, "load_failed",
                commit_id=head.get("commit_id"), errors=errors,
            )
            raise ValueError(
                "No valid checkpoint generation is available. " + " | ".join(errors)
            )

        if not os.path.exists(target):
            return
        if not allow_legacy_unbound:
            self._append_audit(
                target, "load_legacy_unbound_refused",
                legacy_load_policy=LEGACY_LOAD_POLICY,
            )
            raise ValueError(
                "Legacy single-file checkpoint is unbound to the v1.22 "
                "constructor/LR schedule. Refusing automatic load. "
                "Use allow_legacy_unbound=True only after explicit human review."
            )
        data = mx.load(target)
        parts = self._checkpoint_parts(data)
        self._apply_checkpoint(parts)
        self._mark_legacy_unbound_taint(target)
        self._last_checkpoint_event = {
            "operation": "load",
            "status": "legacy-single-file-authorized",
            "generation_id": None,
            "fallback_used": False,
            "config_binding": "legacy-unbound",
            "legacy_load_policy": LEGACY_LOAD_POLICY,
            "explicit_legacy_authorization": True,
        }
        self._append_audit(
            target, "load_legacy_single_file_authorized",
            legacy_load_policy=LEGACY_LOAD_POLICY,
        )

    def count(self) -> int:
        per_layer = self.dim * self.dim + 3 * self.dim
        return 256 * self.dim + self.layercount * per_layer + 256 * self.dim + 256 + self.dim + 1

class Runtime:
    def __init__(self, path: str, threshold: float, **kwargs):
        self.model = Model(**kwargs)
        self.path = path
        self.threshold = threshold

        self.step = 0
        self.prevtime = None

    def save(self):
        self.step += 1
        if self.step % 500 == 0: self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool, save: bool, frozen: bool):
        outputs = self.model(c, n, end, frozen)

        if save: self.save()
        return outputs

    def write(self, b: int):
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, save: bool, frozen: bool):
        while True:
            text = input(f'\n[{self.now()} | {0 if self.prevtime is None else time.time() - self.prevtime:.4f}s]\nUser >> ')
            self.prevtime = time.time()

            data = (text + '\n').encode('utf-8')
            
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2, save, frozen)

            print(f'\n[{self.now()}]\nModel >> ', end = '', flush = True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, save, frozen)
                self.write(b)

                if stop > self.threshold:
                    print()
                    break

    def train(self, save: bool, frozen: bool, dataset: str):
        files = glob.glob(dataset, recursive = True)

        if not files:
            raise FileNotFoundError(
                f'Could not find training files with the following glob: {dataset!r}. Try downloading a dataset first.'
            )

        random.shuffle(files)

        while True:
            for file in files:
                with open(file, 'r', encoding = 'utf-8', errors = 'ignore') as f:
                    for line in f:
                        data = line.encode('utf-8')
                        if len(data) < 2: continue

                        for i, (c, n) in enumerate(itertools.pairwise(data)):
                            b, _ = self.call(c, n, i == len(data) - 2, save, frozen)
                            self.write(b)

    def now(self): return datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

    def __call__(
        self,
        mode: str,
        dataset: str,
        save: bool,
        frozen: bool,
        allow_legacy_unbound: bool = False,
    ):
        self.model.load(
            self.path,
            allow_legacy_unbound=allow_legacy_unbound,
        )
        if save and self.model.checkpoint_authority_status()["legacy_unbound_taint"]:
            self.model._append_audit(
                self.path, "runtime_save_refused_legacy_unbound_taint",
                legacy_unbound_source=self.model._legacy_unbound_source,
            )
            raise ValueError(
                "Legacy-unbound override is inspection/non-authoritative mode. "
                "Use --no-save. Authoritative persistence is refused."
            )
        print()

        try:
            match mode:
                case 'train': self.train(save, frozen, dataset)
                case 'chat': self.chat(save, frozen)

        finally:
            if save: self.model.save(self.path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'test-model-thing')
    parser.add_argument('path')
    parser.add_argument('mode', choices = ['train', 'chat'])

    parser.add_argument('--frozen', action = 'store_true')
    parser.add_argument('--no-save', action = 'store_false')
    parser.add_argument('--dataset', default = 'wikipedia_clean/**/wiki_*')
    parser.add_argument(
        '--allow-legacy-unbound-checkpoint',
        action = 'store_true',
        help = (
            'Explicit human authorization to load a legacy single-file checkpoint '
            'whose constructor/LR schedule cannot be cryptographically bound. '
            'Must be used with --no-save; authoritative persistence is refused.'
        ),
    )

    args = parser.parse_args()

    runtime = Runtime(path = args.path, threshold = 0.35, dim = 512, layers = 16, spread = 32, temp = 0.75, lr = 5e-4, lrbegin = 40000, lrend = 120000)
    print(f'parameters: {runtime.model.count():,}')

    runtime(
        args.mode,
        args.dataset,
        args.no_save,
        args.frozen,
        allow_legacy_unbound = args.allow_legacy_unbound_checkpoint,
    )