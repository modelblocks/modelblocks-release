import glob

def get_basename(s, extension):
    l = len(extension)
    return s[:-l]

def task_hello():
    """hello"""

    def python_hello(targets):
        with open(targets[0], "a") as output:
            output.write("Python says Hello World!!!\n")

    return {
        'actions': [python_hello],
        'targets': ["hello.txt"],
        }

def task_hello_2():
    """hello cmd """
    msg = 3 * "hi! "
    return {
        'actions': ['echo %s ' % msg],
        }

def task_bd_linetrees():
    """hello"""

    pos = '%(pos)s'
    print(pos)

    def deps(pos):
        return pos

    def report(pos):
        return 'echo %s' %pos

    return {
        'actions': [(report,)],
        'file_dep': ['%(pos)s'],
        'pos_arg': 'pos',
        'verbosity': 2,
    }
