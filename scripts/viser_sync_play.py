"""Start playback on several viser servers at the same moment.

Connects to each server's websocket, finds the 'Timestep' slider and 'Playing' checkbox from the
GUI messages the server sends on connect, then sends Timestep=0 and Playing=True to all servers
together. Useful to compare two runs of scripts/visualize_cosilab_viser.py side by side.

python scripts/viser_sync_play.py 8080 8081              # both to timestep 0 and playing
python scripts/viser_sync_play.py 8080 8081 --pause      # pause both
python scripts/viser_sync_play.py 8080 8081 --no-reset   # play from the current timesteps
"""
import argparse
import asyncio
import time

import msgspec.msgpack
import viser
import websockets
import zstandard

_zstd = zstandard.ZstdDecompressor()


def decode_frame(raw):
    """Server -> client frame: [8B inner len][8B compressed len][zstd msgpack][binary buffers]."""
    inner_len = int.from_bytes(raw[:8], 'little')
    comp_len = int.from_bytes(raw[8:16], 'little')
    inner = _zstd.decompress(raw[16:16 + comp_len], max_output_size=inner_len)
    return msgspec.msgpack.decode(inner)['messages']


async def find_controls(port, timeout=20.0):
    ws = await websockets.connect(f'ws://localhost:{port}', subprotocols=[f'viser-v{viser.__version__}'],
                                  max_size=None)
    found = {}
    deadline = time.monotonic() + timeout
    while len(found) < 2:
        raw = await asyncio.wait_for(ws.recv(), max(0.1, deadline - time.monotonic()))
        for m in decode_frame(raw):
            label = m.get('props', {}).get('label') if isinstance(m.get('props'), dict) else None
            if m['type'] == 'GuiSliderMessage' and label == 'Timestep':
                found['step'] = m['uuid']
            elif m['type'] == 'GuiCheckboxMessage' and label == 'Playing':
                found['play'] = m['uuid']
    return ws, found


def update(uuid, value):
    return msgspec.msgpack.encode({'type': 'GuiUpdateMessage', 'uuid': uuid, 'updates': {'value': value}})


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ports', type=int, nargs='+')
    parser.add_argument('--pause', action='store_true', help='pause all instead of playing')
    parser.add_argument('--no-reset', action='store_true', help='do not jump back to timestep 0')
    args = parser.parse_args()

    conns = await asyncio.gather(*(find_controls(p) for p in args.ports))
    for p, (_, f) in zip(args.ports, conns):
        print(f'port {p}: Timestep {f["step"]}, Playing {f["play"]}')

    async def send(ws, f):
        if not args.no_reset and not args.pause:
            await ws.send(update(f['step'], 0))
        await ws.send(update(f['play'], not args.pause))

    await asyncio.gather(*(send(ws, f) for ws, f in conns))
    await asyncio.sleep(0.5)  # let the servers process before the sockets close
    await asyncio.gather(*(ws.close() for ws, _ in conns))
    print('paused' if args.pause else 'playing', 'on ports', args.ports)


if __name__ == '__main__':
    asyncio.run(main())
