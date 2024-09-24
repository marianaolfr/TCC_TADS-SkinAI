import subprocess

def install_packages(libraries_file):
    subprocess.check_call(["pip","install","-r", libraries_file])

if __name__ == "__main__":
    libraries_file = "libraries.txt"
    install_packages(libraries_file)