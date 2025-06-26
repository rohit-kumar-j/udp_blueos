#!/usr/bin/python3

def main():
    from icm20602 import ICM20602
    # from llog import LLogWriter
    import time

    device = "icm20602"
    # parser = LLogWriter.create_default_parser(__file__, device)
    # args = parser.parse_args()

    # with LLogWriter(args.meta, args.output, console=args.console) as log:
    icm = ICM20602()

    for _ in range(100000):
        data = icm.read_all()
        print(f'a.x: {data.a.x} a.y: {data.a.y} a.z: {data.a.z}\n'
                f'g.x: {data.g.x} g.y: {data.g.y} g.z: {data.g.z}\n'
                f't: {data.t}\n')
        time.sleep(0.001)
                # f'{data.a_raw.x} {data.a_raw.y} {data.a_raw.z}\n'
                # f'{data.g_raw.x} {data.g_raw.y} {data.g_raw.z} {data.t_raw}\n\n')


if __name__ == '__main__':
    main()
