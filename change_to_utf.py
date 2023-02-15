import os
import sys
import codecs
import chardet


def convert(filename, out_enc="UTF-8"):
    try:
        # 以只读方式打开文件(r), 二进制文件(b), 打开一个文件进行更新(可读可写 +)
        content = codecs.open(filename, 'rb',errors='ignore').read()
        # 识别打开文件的编码
        source_encoding = chardet.detect(content)['encoding']
        # 获取文件内的内容
        content = content.decode(source_encoding or 'utf8', errors='backslashreplace')
        # 以 UTF-8-SIG 的方式将文件保存
        codecs.open(filename, 'w', encoding=out_enc).write(content)

    except IOError as err:
        print("I/O error:{0}".format(err))


def explore(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            convert(path)


def main():
    explore('./data/')


if __name__ == "__main__":
    main()
