from pathlib import Path
import subprocess
from typing import List, Optional, Dict

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class CommandLineException(BaseException):
    def __init__(self, cmd, exit_code):
        super().__init__("`{}` encountered an error. (code: {})".format(cmd, exit_code))
        self.cmd = cmd
        self.exit_code = exit_code


def call_command(command: str,
                 args: List[str],
                 cwd: Path = None,
                 shell: bool = False,
                 environment: Optional[Dict[str, str]] = None,
                 output_path: Path = None) -> int:
    """
    Executes the command (using the subprocess module).
    :param command: The binary to run.
    :param args: The command-line arguments.
    :param cwd: The `cwd param in subprocess. If not `None`, the function changes
    the working directory to cwd prior to execution.
    :param shell: Indicates whether or not to instantiate a shell from which to invoke the command (not recommended!)
    :param environment: A key-value pair representing an environment with necessary variables set.
    :param output_path: A path to print the contents of STDOUT to. (If None, logs STDOUT instead.)
    :return: The exit code. (zero by default, the program's returncode if error.)
    """
    args = [str(arg) for arg in args]

    logger.debug("EXECUTE {cwdstr}{cmd} {arguments}".format(
        cmd=command,
        cwdstr="" if cwd is None else "[cwd={}] ".format(cwd),
        arguments=" ".join(args)
    ))

    if environment is not None:
        logger.debug("ENV: \n{}".format(
            "\n".join(
                f"{key}: {value}" for key, value in environment.items()
            )
        ))

    if output_path is not None:
        logger.debug("STDOUT redirect to {}.".format(output_path))
        output_file = open(output_path, 'w')
    else:
        output_file = None

    try:
        p = subprocess.run(
            [command] + args,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=output_file if output_file is not None else subprocess.PIPE,
            shell=shell,
            cwd=None if cwd is None else str(cwd),
            env=environment
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Encountered file error running subprocess. Is `{command}` installed?") from e

    if output_file is not None:
        output_file.close()

    stdout_bytes = p.stdout
    stderr_bytes = p.stderr

    if stdout_bytes is not None and len(stdout_bytes) > 0:
        logger.debug("STDOUT: {}".format(stdout_bytes.decode("utf-8").strip()))
    if stderr_bytes is not None and len(stderr_bytes) > 0:
        logger.debug("STDERR: {}".format(stderr_bytes.decode("utf-8").strip()))

    return p.returncode
