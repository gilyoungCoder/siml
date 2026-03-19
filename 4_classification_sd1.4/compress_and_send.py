import tarfile
import argparse
import os
import subprocess


def arg_parser():
    parser = argparse.ArgumentParser(description='Compress and send files')
    parser.add_argument('file_match', type=str, help='Files to compress and send')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    return parser.parse_args()

def compress_and_send(files, output):
    file_dir = os.path.dirname(files)
    file_name = os.path.basename(files)

    print(file_dir)
    print

    compressed_files = []
    for file_elem in os.listdir(file_dir):
        if file_elem.startswith(file_name):
            compressed_files.append(file_elem)
    
    if output is None:
        output = os.path.join(file_dir, file_name + ".tar.gz")

    with tarfile.open(output, "w:gz") as tar:
        for file_elem in compressed_files:
            print(f"Compressing {file_elem}")
            tar.add(os.path.join(file_dir, file_elem), arcname=file_elem)

    if os.path.exists("/home/djfelrl11/awesome_scripts/gcp/gcp_connect_to_siml.sh"):
        print("Connecting to SIML")
        subprocess.run(["bash", "/home/djfelrl11/awesome_scripts/gcp/gcp_connect_to_siml.sh"])

    print("Sending to SIML")
    scp_script = []
    scp_script.append("scp -i /home/djfelrl11/.ssh/google_compute_engine -P 2022")
    scp_script.append(output)
    scp_script.append("jeongjun@137.68.191.45:/mnt/home/jeongjun/layout_diffusion/additional_obj_det_models/datasets")
    subprocess.run(" ".join(scp_script), shell=True)



if __name__ == "__main__":
    args = arg_parser()
    compress_and_send(args.file_match, args.output)