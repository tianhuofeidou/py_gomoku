# -*- coding: utf-8 -*-
"""把桌面版纯算法引擎同步到 Android 工程的内嵌 Python 目录。

规则：
- 唯一事实来源是仓库根目录的 gomoku/；
- 只同步移动端需要的文件，不同步 tests/nn/cli/adapter；
- tools/ga_tune.py 必须使用移动端占位实现，避免引入桌面 GA 调参链路；
- protocol.py（Gomocup 管道协议入口）属于桌面/Gomocup 侧能力，不在同步清单内，
  移动端通过自己的界面与 android_api.py 驱动引擎；
- android/app/src/main/python/android_api.py 是 Android 专用文件，本脚本不处理。

用法：
    python scripts/sync_android_engine.py          # 同步
    python scripts/sync_android_engine.py --check  # 只检查是否已同步，CI 用
"""
import argparse
import sys
from pathlib import Path

# Windows CI 控制台默认可能是 cp1252；统一按 UTF-8 输出，避免打印中文时
# 抛出 UnicodeEncodeError。旧 Python 没有 reconfigure 时保持原样。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# 桌面引擎中需要复制到 Android 的顶层文件
TOP_FILES = (
    '__init__.py',
    'game.py',
    'loss_book.py',      # game._archive_finished 会 import 它，漏了会在 Android 上 ImportError
    'memory.py',
    'paths.py',
    'service.py',
    'session.py',
)

# 桌面引擎中需要复制到 Android 的 core 文件。
# 注意：gomoku/core/evaluate.py 被 .gitignore 排除，GitHub 干净检出不存在；
# 移动端也不启用神经网络，因此不纳入同步清单。
CORE_FILES = (
    '__init__.py',
    'utils.py',
    'pattern.py',
    'search.py',
    'deep_search.py',
    'opening_book.py',
    'opening_book_data.py',
    'engine.py',
    'train.py',
)

# Android 端 tools/ga_tune.py 的占位实现。
# service.load_ga_params 会导入该模块；移动端不启用 GA 调参，因此只保留接口。
GA_TUNE_STUB = '''# -*- coding: utf-8 -*-
"""Android 端 GA 参数加载占位。

桌面版的 tools/ga_tune.py 依赖 multiprocessing 等桌面调参链路；
移动端只运行纯算法引擎，不启用 GA 调参（需要 DSH_GOMOKU_GA=1）。
保留 service.load_ga_params 需要的两个接口，行为为无操作。
"""


def _fix_genome(params):
    return params


def apply_params(params):
    return None
'''


def repo_root():
    return Path(__file__).resolve().parents[1]


def expected_files(root):
    """返回 {相对 gomoku 的路径: 内容 bytes}。tools/ga_tune.py 使用占位内容。"""
    src = root / 'gomoku'
    out = {}
    for name in TOP_FILES:
        out[Path(name)] = (src / name).read_bytes()
    for name in CORE_FILES:
        out[Path('core') / name] = (src / 'core' / name).read_bytes()
    out[Path('tools') / '__init__.py'] = (src / 'tools' / '__init__.py').read_bytes()
    out[Path('tools') / 'ga_tune.py'] = GA_TUNE_STUB.encode('utf-8')
    return out


def dest_root(root):
    return root / 'android' / 'app' / 'src' / 'main' / 'python' / 'gomoku'


def _normalize_newlines(data):
    """Git 在 Windows 检出时可能把 LF 转换为 CRLF；比较时统一按 LF 归一化。"""
    return data.replace(b'\r\n', b'\n')


def _module_level_internal_imports(py_path):
    """提取文件内**所有层级**对 gomoku 内部模块的 import，返回清单相对路径集合。

    为什么连函数体内的延迟 import 也要查：``game._archive_finished`` 里用
    ``from gomoku import loss_book`` 是延迟 import，同步漏了它虽然被 try/except
    兜住（不会崩），但 Android 端会**静默失去收录能力**——这正是要防的缺口。
    ``gomoku.tools.*`` 不在 Android 同步范围（只有 ga_tune 占位），单独排除。
    """
    import re
    try:
        text = py_path.read_text(encoding='utf-8')
    except Exception:
        return set()
    out = set()
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith('from gomoku'):
            continue
        m = re.match(r'from\s+gomoku\.(\w+)(?:\.(\w+))?\s+import\b', s)
        if m:
            pkg, mod = m.group(1), m.group(2)
            # tools 与 nn 都不在 Android 同步范围：
            #   tools/ 只有 ga_tune 占位；nn/ 是本地实验代码（被 .gitignore 排除）
            if pkg in ('tools', 'nn'):
                continue
            if mod:
                out.add('%s/%s.py' % (pkg, mod))
            continue
        m = re.match(r'from\s+gomoku\s+import\s+(.+)$', s)
        if m:
            for name in m.group(1).split('#')[0].split(','):
                name = name.strip().split(' as ')[0].strip()
                if name.isidentifier():
                    out.add(name + '.py')
    return out


def check_import_closure(src, expected):
    """检查"清单内的文件在模块级 import 了清单外的内部模块"。

    这类缺口会让同步"通过"但 Android 端 ImportError——
    game.py 引用 loss_book.py 就是这么漏出去的。
    """
    known = {str(p).replace('\\', '/') for p in expected}
    problems = []
    for rel in sorted(known):
        if not rel.endswith('.py'):
            continue
        path = src / rel
        if not path.is_file():
            continue
        for dep in sorted(_module_level_internal_imports(path)):
            if dep not in known:
                problems.append('%s 模块级 import 了清单外的 %s' % (rel, dep))
    return problems


def check(root):
    """返回 (errors, warnings)。errors 非空表示同步漂移。"""
    src = root / 'gomoku'
    dst = dest_root(root)
    errors = []
    warnings = []
    if not src.is_dir():
        return [f'桌面引擎目录不存在: {src}'], []
    if not dst.is_dir():
        return [f'Android 引擎目录不存在: {dst}'], []

    expected = expected_files(root)
    errors.extend(check_import_closure(src, expected))
    for rel, want in expected.items():
        target = dst / rel
        if not target.is_file():
            errors.append(f'缺少文件: {rel}')
            continue
        if _normalize_newlines(target.read_bytes()) != _normalize_newlines(want):
            errors.append(f'内容不同步: {rel}')

    known = {str(p).replace('\\', '/') for p in expected}
    for path in dst.rglob('*.py'):
        rel = path.relative_to(dst).as_posix()
        if rel not in known:
            errors.append(f'存在未纳入同步清单的文件: {rel}')
    return errors, warnings


def sync(root):
    """按清单写入 Android 内嵌引擎，然后返回检查结果。"""
    dst = dest_root(root)
    expected = expected_files(root)
    for rel, data in expected.items():
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and _normalize_newlines(target.read_bytes()) == _normalize_newlines(data):
            continue
        target.write_bytes(data)
    return check(root)


def main():
    parser = argparse.ArgumentParser(description='同步桌面纯算法引擎到 Android 工程')
    parser.add_argument('--check', action='store_true',
                        help='只检查是否已同步，不写入文件')
    args = parser.parse_args()

    root = repo_root()
    errors, warnings = check(root) if args.check else sync(root)
    action = '检查' if args.check else '同步'
    if errors:
        print(f'[{action}] 失败：发现 {len(errors)} 处不同步')
        for item in errors:
            print('  - ' + item)
        return 1

    print(f'[{action}] 通过：Android 内嵌引擎与桌面 gomoku/ 一致')
    for item in warnings:
        print('  ! ' + item)
    return 0


if __name__ == '__main__':
    sys.exit(main())
