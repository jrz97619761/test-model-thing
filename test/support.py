import random


def _array_values(array):
    import mlx.core as mx

    mx.eval(array)
    return [float(value) for value in array.tolist()]


def _flatten_values(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten_values(item)
    else:
        yield float(value)


def _gradient_summary(array):
    values = list(_flatten_values(array.tolist()))
    return {
        'sum': sum(values),
        'sum_squares': sum(value * value for value in values),
        'max_abs': max((abs(value) for value in values), default=0.0),
    }


def capture_rtu_baseline(config, seed, steps, output_indices, state_rows):
    import mlx.core as mx
    from tmt.main import Model

    random.seed(seed)
    mx.random.seed(seed)
    model = Model(**config)
    mx.eval(model.parameters())

    updates = []
    captured_gradients = []
    original_updategrads = model.updategrads

    def capture_updategrads(grads):
        gradients = {
            f'blocks.{index}.decay': _array_values(block_grads['decay'])
            for index, block_grads in enumerate(grads['blocks'])
        }
        encoder_gradient = grads['encoder']['embed']['weight']
        mx.eval(encoder_gradient)
        gradients['encoder.embed.weight'] = _gradient_summary(encoder_gradient)
        gradients['decoder.stop.bias'] = _array_values(grads['decoder']['stop']['bias'])
        captured_gradients.append(gradients)
        original_updategrads(grads)
        mx.eval(model.parameters(), model.optimizer.state)

    model.updategrads = capture_updategrads

    for step in steps:
        record = {'input': step['input'], 'target': step['target'], 'end': step['end']}
        updates.append(record)
        metrics = {}
        output, stop, loss = model.train_step(
            step['input'], step['target'], step['end'], metrics=metrics
        )
        mx.eval(output, stop, loss, *metrics.values())
        record['logits'] = {
            str(index): float(output[index].item()) for index in output_indices
        }
        record['prediction'] = int(mx.argmax(output).item())
        record['stop'] = float(stop.item())
        record['loss'] = float(loss.item())
        record['components'] = {name: metrics[name] for name in sorted(metrics)}
        record['gradients'] = captured_gradients.pop()
        record['state'] = {
            f'blocks.{index}': {
                'states': _array_values(layer.states),
                'decaytrace': _array_values(layer.decaytrace),
                'embedtrace_rows': {
                    str(byte): _array_values(layer.embedtrace[byte])
                    for byte in state_rows
                },
            }
            for index, layer in enumerate(model.blocks)
        }

    return updates
