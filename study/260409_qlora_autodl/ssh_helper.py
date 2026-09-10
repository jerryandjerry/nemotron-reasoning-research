"""SSH helper for AutoDL instances."""
import paramiko
import sys

INSTANCES = {
    '5090': {'host': 'connect.bjb2.seetacloud.com', 'port': 45768, 'user': 'root', 'password': dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['SSH_PASSWORD']},
    '4090': {'host': 'connect.bjb1.seetacloud.com', 'port': 28591, 'user': 'root', 'password': dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['SSH_PASSWORD']},
}

def run(instance_key, cmd, timeout=120):
    """Run command on instance, return stdout."""
    info = INSTANCES[instance_key]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info['host'], port=info['port'], username=info['user'],
                   password=info['password'], timeout=15)
    stdin, stdout, stderr = client.exec_command(
        f"export PATH=/root/miniconda3/bin:$PATH && {cmd}", timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    client.close()
    return out, err

def upload(instance_key, local_path, remote_path):
    """Upload file via SFTP."""
    info = INSTANCES[instance_key]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info['host'], port=info['port'], username=info['user'],
                   password=info['password'], timeout=15)
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    client.close()
