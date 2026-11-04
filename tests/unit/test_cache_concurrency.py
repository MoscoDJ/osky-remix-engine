from concurrent.futures import ProcessPoolExecutor
from osky_remix.core.cache import CacheStore, cache_key


def publish(root):
    cache = CacheStore(root)
    key = cache_key('race', '1', [])
    for _ in range(8):
        cache.put_json(key, {'values':list(range(100))})
        assert cache.get_json(key) == {'values':list(range(100))}
    return key


def test_process_writers_do_not_remove_each_other(tmp_path):
    with ProcessPoolExecutor(max_workers=3) as pool:
        keys = list(pool.map(publish, [tmp_path]*3))
    assert len(set(keys)) == 1
    assert CacheStore(tmp_path).get_json(keys[0])['values'] == list(range(100))
