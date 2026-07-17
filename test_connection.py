import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Test connection to Carla Simulator.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Carla host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=2000, help="Carla port (default: 2000)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout in seconds")
    
    args = parser.parse_args()
    
    try:
        import carla
    except ImportError:
        print("Error: The 'carla' python package is not installed.")
        print("Please install it (e.g. 'pip install carla') or add it to your python path.")
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
    except Exception as e:
        print(f"Failed to connect to Carla: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
