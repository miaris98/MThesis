import argparse
import sys
import os
import time

def main():
    parser = argparse.ArgumentParser(description="Test connection to Carla Simulator.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Carla host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=2000, help="Carla port (default: 2000)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Carla connection timeout in seconds")
    parser.add_argument("--restart", action="store_true", help="Auto-restart CARLA server if connection fails or requested")
    parser.add_argument("--town", type=str, default="Town10HD_Opt", help="Map to load on restart")
    
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.config.carla_client import add_carla_to_path, carla_root
    root = carla_root()
    log_dir = os.path.dirname(root)
    if args.restart and os.path.exists(f"{root}/CarlaUE4.sh"):
        print(f"--> Restarting CARLA server session on map {args.town}...")
        os.system("pkill -9 -f CarlaUE4 2>/dev/null || true")
        os.system(f"chmod 1777 /tmp; chmod 700 /tmp/tmux-*; chown root:root /tmp/tmux-0; chmod -R 777 {root} /dev/dri; chmod -R 666 /dev/nvidia* 2>/dev/null || true")
        os.system(f"tmux new-session -d -s carla_server \"su carlauser -c 'export SDL_VIDEODRIVER=offscreen; {root}/CarlaUE4.sh /Game/Carla/Maps/{args.town} -carla-rpc-port={args.port} -port={args.port} -RenderOffScreen -nosound -quality-level=Low' > {log_dir}/carla_server.log 2>&1\"")
        time.sleep(10)

    add_carla_to_path(with_agents=False)

    try:
        import carla
    except ImportError as e:
        print(f"Error importing carla: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your python version and CARLA_ROOT path.")
        sys.exit(1)
        
    print(f"Connecting to Carla simulator at {args.host}:{args.port}...")
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        server_version = client.get_server_version()
        client_version = client.get_client_version()
        print(f"Successfully connected to Carla Simulator!")
        print(f"Client Version: {client_version}")
        print(f"Server Version: {server_version}")
        
        # Test getting world
        world = client.get_world()
        map_name = world.get_map().name
        print(f"Current Map: {map_name}")
        try:
            available_maps = client.get_available_maps()
            print(f"Available Maps on Server ({len(available_maps)}): {', '.join(available_maps)}")
        except Exception:
            pass
    except Exception as e:
        print(f"Failed to connect to Carla: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
