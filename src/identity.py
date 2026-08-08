def pick(row, *names, default=None):
    for name in names:
        if name in row and row[name] not in ('',None): return row[name]
    return default

def stable_pk(video_id: str, keyframe_n: int) -> int:
    group=video_id.split('_')[0]
    return int(group[1:])*10**12 + int(video_id.split('_V')[-1])*10**7 + int(keyframe_n)
