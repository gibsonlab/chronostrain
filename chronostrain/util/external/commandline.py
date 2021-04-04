import subprocess
from typing import List
from chronostrain.util import logger


class CommandLineException(BaseException):
    def __init__(self, cmd, exit_code):
        super().__init__("`{}` encountered an error.".format(cmd))
        self.cmd = cmd
        self.exit_code = exit_code


def call_command(command: str,
                 args: List[str],
                 cwd: str = None,
                 output_path: str = None) -> int:
    """
    Executes the command (using the subprocess module).
    :param command: The binary to run.
    :param args: The command-line arguments.
    :param cwd: The `cwd param in subprocess. If not `None`, the function changes
    the working directory to cwd prior to execution.
    :param output_path: A path to print the contents of STDOUT to. (If None, logs STDOUT instead.)
    :return: The exit code. (zero by default, the program's returncode if error.)
    """
    logger.debug("EXECUTE {}: {} {}".format(
        command,
        "" if cwd is None else "[cwd={}]".format(cwd),
        " ".join(args)
    ))

    if output_path is None:
        p = subprocess.run(
            [command] + args,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=cwd
        )
        logger.debug("STDOUT: {}".format(p.stdout.decode("utf-8")))
        logger.debug("STDERR: {}".format(p.stderr.decode("utf-8")))
    else:
        with open(output_path, 'w') as outfile:
            p = subprocess.run(
                [command] + args,
                stdout=outfile,
                cwd=cwd
            )
        logger.debug("STDOUT saved to {}.".format(output_path))
        logger.debug("STDERR: {}".format(p.stderr.decode("utf-8")))
    return p.returncode
