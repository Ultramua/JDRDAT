import sys


class Tee:
    def __init__(self, fname, mode="a"):
        # 这两行代码将类的实例变量stdout设置为标准输出（即通常打印到控制台的内容），并打开一个文件并将其赋值给实例变量file。
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        # 这几行代码将message写入到标准输出和文件中。self.stdout.write(message)将message写入标准输出，
        # 而self.file.write(message)将message写入文件中。
        # 然后，self.flush()方法用于刷新输出，确保写入文件的内容被立即保存
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()