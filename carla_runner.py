import os
import subprocess
import time
import sys
import argparse

class CarlaRunner:
    def __init__(self, carla_path=None, port=2000, headless=True, graphics_api="vulkan", quality_level="Low"):
        self.carla_path = carla_path or self._detect_carla_path()
        self.port = port
        self.headless = headless
        self.graphics_api = graphics_api.lower()
        self.quality_level = quality_level
        self.process = None

    def _detect_carla_path(self):
        # Check environment variable
        carla_root = os.environ.get("CARLA_ROOT")
        if carla_root:
            exe_path = os.path.join(carla_root, "CarlaUE4.exe")
            if os.path.exists(exe_path):
                return exe_path
            # Maybe it's a Linux package structure or other
            exe_path_bin = os.path.join(carla_root, "CarlaUE4", "Binaries", "Win64", "CarlaUE4-Win64-Shipping.exe")
            if os.path.exists(exe_path_bin):
                return exe_path_bin

        # Common Windows installation folders (including external drives)
        possible_dirs = [
            r"C:\Carla",
            r"C:\carla",
            r"D:\Carla",
            r"D:\carla",
            r"E:\Carla",
            r"E:\carla",
            r"F:\Carla",
            r"F:\carla",
            r"G:\Carla",
            r"G:\carla",
            os.path.expanduser("~/Carla"),
        ]
        
        for pdir in possible_dirs:
            if os.path.exists(pdir):
                # Search for CarlaUE4.exe recursively or directly
                for root, dirs, files in os.walk(pdir):
                    if "CarlaUE4.exe" in files:
                        return os.path.join(root, "CarlaUE4.exe")
                    if "CarlaUE4-Win64-Shipping.exe" in files:
                        return os.path.join(root, "CarlaUE4-Win64-Shipping.exe")

        return None

    def start(self):
        if not self.carla_path or not os.path.exists(self.carla_path):
            raise FileNotFoundError(
                f"Carla executable not found. Please set CARLA_ROOT environment variable "
                f"or pass the correct path. Attempted path: {self.carla_path}"
            )

        cmd = [self.carla_path]
        
        # Port
        cmd.append(f"-carla-port={self.port}")
        
        # Headless mode
        if self.headless:
            cmd.append("-RenderOffScreen")
            
        # Graphics API
        if self.graphics_api == "vulkan":
            cmd.append("-vulkan")
        elif self.graphics_api == "dx12" or self.graphics_api == "d3d12":
            cmd.append("-dx12")
        elif self.graphics_api == "opengl":
            cmd.append("-opengl")

        # Quality Level
        if self.quality_level:
            cmd.append(f"-quality-level={self.quality_level}")

        print(f"Launching Carla Simulator with command: {' '.join(cmd)}")
        
        # Launching subprocess
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        
        # Give it a few seconds to initialize
        time.sleep(5)
        
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(
                f"Carla failed to start. Exit code: {self.process.returncode}\n"
                f"Stdout: {stdout.decode(errors='ignore')}\n"
                f"Stderr: {stderr.decode(errors='ignore')}"
            )
        
        print(f"Carla Simulator started successfully on port {self.port} (PID: {self.process.pid})")

    def stop(self):
        if self.process:
            print(f"Terminating Carla Simulator (PID: {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process did not terminate in time. Killing it...")
                self.process.kill()
            self.process = None
            print("Carla Simulator stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Carla Simulator execution.")
    parser.add_argument("--path", type=str, help="Path to CarlaUE4 executable")
    parser.add_argument("--port", type=int, default=2000, help="Carla port (default: 2000)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless offscreen mode")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Run with a visual window")
    parser.add_argument("--graphics", type=str, default="vulkan", choices=["vulkan", "dx12", "opengl"], help="Graphics API to use")
    parser.add_argument("--quality", type=str, default="Low", choices=["Low", "Epic"], help="Quality level")

    args = parser.parse_args()
    
    runner = CarlaRunner(
        carla_path=args.path,
        port=args.port,
        headless=args.headless,
        graphics_api=args.graphics,
        quality_level=args.quality
    )
    
    try:
        runner.start()
        print("Press Ctrl+C to stop the simulator...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        runner.stop()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
