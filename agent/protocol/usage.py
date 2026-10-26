"""One UI usage shape for live replies and persisted provider snapshots."""

def normalize_usage(value):
    if not isinstance(value, dict):
        return None
    def count(*keys):
        for key in keys:
            if key in value:
                try:
                    return max(0, int(value[key] or 0))
                except (ValueError, TypeError, OverflowError):
                    return 0
        return 0
    inp = count('input', 'prompt_tokens')
    out = count('output', 'completion_tokens')
    hit = min(inp, count('cache_hit', 'prompt_cache_hit_tokens'))
    return dict(input=inp, output=out, total=inp + out,
                prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out,
                prompt_cache_hit_tokens=hit, prompt_cache_miss_tokens=inp - hit,
                cache_hit=hit, cache_miss=inp - hit,
                calls=count('calls'), scope=value.get('scope', 'own'))


def team_usage(own, children):
    """Add only direct child totals; an inclusive snapshot is already summed."""
    result = normalize_usage(own)
    if result and result['scope'] == 'team':
        return result
    for child in children:
        child = normalize_usage(child)
        if child is None:
            continue
        if result is None:
            result = normalize_usage({})
        for key in ('input', 'output', 'total', 'cache_hit', 'cache_miss', 'calls'):
            result[key] += child[key]
    if result is not None:
        result['scope'] = 'team'
    return normalize_usage(result)


def own_usage(snapshot, children):
    """Display only this agent; migrate previously stored inclusive snapshots."""
    result = normalize_usage(snapshot)
    if result is None:
        return None
    if result['scope'] == 'team':
        for child in children:
            child = normalize_usage(child)
            if child is None:
                continue
            for key in ('input', 'output', 'total', 'cache_hit', 'cache_miss', 'calls'):
                result[key] = max(0, result[key] - child[key])
    result['scope'] = 'own'
    return normalize_usage(result)
