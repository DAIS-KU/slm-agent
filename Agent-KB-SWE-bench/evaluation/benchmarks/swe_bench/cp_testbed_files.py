import json
import os
import subprocess

# ------------------- 配置区 -------------------
LOCAL_SAVE_DIR = './testbed'  # 本地保存目录
# ---------------------------------------------


def get_docker_images():
    """获取所有Docker镜像列表"""
    try:
        output = (
            subprocess.check_output(
                ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .splitlines()
        )
        return [img for img in output if img != '<none>:<none>']
    except subprocess.CalledProcessError as e:
        print(f'❌ 获取镜像失败: {e.output.decode().strip()}')
        exit(1)


def select_image(images):
    """用户选择镜像"""
    print('\n可用Docker镜像:')
    selected_images = []
    for i, img in enumerate(images):
        if img.startswith('swebench'):
            selected_images.append(img)
    print(f'selected_images: {selected_images}')
    return selected_images
    # while True:
    #     try:
    #         # choice = int(input("请选择要操作的镜像编号: ")) - 1
    #         # if 0 <= choice < len(images):
    #         #     return images[choice]
    #         # print("编号超出范围，请重新输入")
    #     except ValueError:
    #         print("请输入有效数字")


def start_container(image):
    """启动Docker容器并保持运行"""
    container_name = f"temp_{image.replace('/', '_').replace(':', '_')}"
    try:
        subprocess.check_output(
            [
                'docker',
                'run',
                '-d',
                '--name',
                container_name,
                image,
                'tail',
                '-f',
                '/dev/null',
            ],
            stderr=subprocess.STDOUT,
        )
        return container_name
    except subprocess.CalledProcessError as e:
        print(f'❌ 启动容器失败: {e.output.decode().strip()}')
        exit(1)


def find_report_files(img):
    """查找容器内的report.json文件"""
    report_path = os.path.join(REPORT_DIR_PATH, img, 'report.json')
    test_files = {'all': []}
    with open(report_path, 'r') as f:
        report_data = json.load(f)
        tests_status = report_data[img]['tests_status']
        for k, v in tests_status.items():
            test_files['all'].extend(v['success'])
            test_files['all'].extend(v['failure'])
            test_files[k] = v
    print(f'已找到报告文件: {test_files}')
    return test_files


def process_reports(case_name, container_name, test_files):
    """处理报告文件并复制测试文件"""
    local_path = os.path.join(LOCAL_SAVE_DIR, case_name)
    os.makedirs(local_path, exist_ok=True)

    # 复制文件到本地
    target_file_paths = set()
    for file_path in test_files['all']:
        parsed_file_path = file_path.split('::')[0]
        img_file_path = os.path.join('testbed', parsed_file_path)
        target_file_paths.add(img_file_path)

    for file_path in target_file_paths:
        print(f'正在复制文件: {img_file_path}')
        subprocess.run(
            ['docker', 'cp', f'{container_name}:{img_file_path}', local_path],
            check=True,
        )
        print(f'已复制: {file_path} -> {os.path.abspath(local_path)}')
    with open(os.path.join(local_path, 'report.json'), 'w') as f:
        json.dump(test_files, f, indent=4)


def cleanup(container_name):
    """清理容器"""
    subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)


def main():
    # 获取并选择镜像
    images = get_docker_images()
    if not images:
        print('没有找到可用Docker镜像')
        return
    selected_image = select_image(images)

    # 启动容器
    for img in selected_image:
        print(f'🔄 启动容器: {img}')
        container_name = start_container(img)
        print(f'🔄 已启动临时容器: img: {img}, container: {container_name}')
        tmp = img.split(':')[0].split('.')[-1].split('_')
        case_name = '__'.join([tmp[0], tmp[2]])
        print(f'🔄 查找报告文件: {case_name}')
        try:
            # 查找报告文件
            test_files = find_report_files(case_name)
            # 处理报告并复制文件
            process_reports(case_name, container_name, test_files)
        except:
            print(f'❌ 处理报告文件失败: {case_name}')
        finally:
            # 清理容器
            cleanup(container_name)
            print(f'🧹 已清理容器: {container_name}')


if __name__ == '__main__':
    main()
