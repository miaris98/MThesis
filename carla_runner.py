import os
import subprocess
import time
import sys
import argparse

class CarlaRunner:
    def __init__(self, carla_path=None, port=2000, headless=True, graphics_api="vulkan", quality_level="Low", use_docker=False, docker_image="carlasim/carla:0.9.15"):
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.port = port
        self.headless = headless
        self.graphics_api = graphics_api.lower()
        self.quality_level = quality_level
        self.carla_path = carla_path or (None if use_docker else self._detect_carla_path())
        self.process = None
        self.container_name = f"carla_server_{self.port}"

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
                for root, dirs, files in os.walk(pdir):
                    if "CarlaUE4.exe" in files:
                        return os.path.join(root, "CarlaUE4.exe")
                    if "CarlaUE4-Win64-Shipping.exe" in files:
                        return os.path.join(root, "CarlaUE4-Win64-Shipping.exe")

        return None

    def start(self):
        if self.use_docker:
            self._start_docker()
        else:
            self._start_native()

    def _start_docker(self):
        print(f"Launching Carla Simulator container ({self.docker_image})...")
        
        # Stop existing container if running with same name
        subprocess.run(["docker", "rm", "-f", self.container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        p_start = self.port
        p_end = self.port + 2
        
        cmd = [
            "docker", "run",
            "-d",
            "--name", self.container_name,
            "--rm",
            "--gpus", "all",
            "-p", f"{p_start}-{p_end}:{p_start}-{p_end}",
            self.docker_image,
            "./CarlaUE4.sh",
            f"-carla-port={self.port}"
        ]

        if self.headless:
            cmd.append("-RenderOffScreen")

        if self.graphics_api == "vulkan":
            cmd.append("-vulkan")
        elif self.graphics_api == "opengl":
            cmd.append("-opengl")

        if self.quality_level:
            cmd.append(f"-quality-level={self.quality_level}")

        print(f"Executing: {' '.join(cmd)}")

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Carla Docker container failed to start: {res.stderr}")

        print("Waiting 12 seconds for CARLA engine & shaders to initialize in Docker...")
        time.sleep(12)
        print(f"Carla Simulator Docker container '{self.container_name}' started successfully on ports {p_start}-{p_end}!")

    def _start_native(self):
        if not self.carla_path or not os.path.exists(self.carla_path):
            raise FileNotFoundError(
                f"Carla executable not found. Please set CARLA_ROOT environment variable "
                f"or pass the correct path. Attempted path: {self.carla_path}"
            )

        cmd = [self.carla_path, f"-carla-port={self.port}"]
        
        if self.headless:
            cmd.append("-RenderOffScreen")
            
        if self.graphics_api == "vulkan":
            cmd.append("-vulkan")
        elif self.graphics_api in ["dx12", "d3d12"]:
            cmd.append("-dx12")
        elif self.graphics_api == "opengl":
            cmd.append("-opengl")

        if self.quality_level:
            cmd.append(f"-quality-level={self.quality_level}")

        print(f"Launching native Carla Simulator: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        
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
        if self.use_docker:
            print(f"Stopping Docker container '{self.container_name}'...")
            subprocess.run(["docker", "stop", self.container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if self.process:
                self.process.terminate()
                self.process = None
            print("Docker container stopped.")
        else:
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
    parser.add_argument("--path", type=str, help="Path to CarlaUE4 executable (for native execution)")
    parser.add_argument("--port", type=int, default=2000, help="Carla port (default: 2000)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless offscreen mode")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Run with a visual window")
    parser.add_argument("--graphics", type=str, default="vulkan", choices=["vulkan", "dx12", "opengl"], help="Graphics API to use")
    parser.add_argument("--quality", type=str, default="Low", choices=["Low", "Epic"], help="Quality level")
    parser.add_argument("--docker", action="store_true", help="Run CARLA inside Docker container")
    parser.add_argument("--image", type=str, default="carlasim/carla:0.9.15", help="Docker image name")

    args = parser.parse_args()
    
    runner = CarlaRunner(
        carla_path=args.path,
        port=args.port,
        headless=args.headless,
        graphics_api=args.graphics,
        quality_level=args.quality,
        use_docker=args.docker,
        docker_image=args.image
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
