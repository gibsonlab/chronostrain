from pathlib import Path
import subprocess
from typing import List, Optional, Dict, Any, Union, Tuple, TextIO
from io import TextIOBase

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class CommandLineException(Exception):
    def __init__(self, cmd, exit_code, stderr_msg: str = ''):
        super().__init__("`{}` encountered an error. (code: {})".format(cmd, exit_code))
        self.cmd = cmd
        self.exit_code = exit_code
        self.stderr_msg = stderr_msg


def call_command(command: str,
                 args: List[Any],
                 cwd: Path = None,
                 shell: bool = False,
                 environment: Optional[Dict[str, str]] = None,
                 stdout: Union[None, Path, TextIOBase] = None,
                 stdin: int = subprocess.PIPE,
                 stderr: int = subprocess.PIPE,
                 silent: bool = False,
                 piped_command: Optional[str] = None) -> int:
    """
    Executes the command (using the subprocess module).
    :param command: The binary to run.
    :param args: The command-line arguments.
    :param cwd: The `cwd param in subprocess. If not `None`, the function changes
    the working directory to cwd prior to execution.
    :param shell: Indicates whether to instantiate a shell from which to invoke the command (not recommended!)
    :param environment: A key-value pair representing an environment with necessary variables set.
    :param stdout: Specifies where to direct the contents of STDOUT to.
    If None, does nothing with output (other than logging, unless silent = True which suppresses all logging).
    If a file path is specified, dumps the contents onto the file.
    If a StringIO object is specified, dumps the contents into this stream.
    :param stdin: (default: subprocess.PIPE)
    :param stderr: (default: subprocess.PIPE)
    :param silent: Determines whether to send debug messages to the logger.
    :return: The exit code. (zero by default, the program's returncode if error.)
    """
    args = [str(arg) for arg in args]

    if not silent:
        if piped_command is None:
            logger.debug("EXECUTE {cwdstr}{cmd} {arguments}".format(
                cmd=command,
                cwdstr="" if cwd is None else "[cwd={}] ".format(cwd),
                arguments=" ".join(args)
            ))
        else:
            logger.debug("EXECUTE {cwdstr}{cmd} {arguments} | {next_cmd}".format(
                cmd=command,
                cwdstr="" if cwd is None else "[cwd={}] ".format(cwd),
                arguments=" ".join(args),
                next_cmd=piped_command
            ))

    if stdout is None:
        result = _run_subprocess(command, args, cwd, shell, environment, subprocess.PIPE, stdin, stderr, piped_command)
    elif isinstance(stdout, Path):
        if not silent:
            logger.debug("STDOUT redirect to {}.".format(stdout))
        with open(stdout, 'wt') as f:
            result = _run_subprocess(command, args, cwd, shell, environment, f, stdin, stderr, piped_command)
    elif isinstance(stdout, TextIOBase):
        # Some other arbitrary textio (which might not have a fileno)
        result = _run_subprocess(command, args, cwd, shell, environment, subprocess.PIPE, stdin, stderr, piped_command)
        stdout.write(result[0].decode('utf-8'))
    else:
        raise RuntimeError("Unsupported `stdout` argument type: {}".format(type(stdout)))

    stdout_bytes, stderr_bytes, returncode = result
    if not silent:
        if stdout_bytes is not None and len(stdout_bytes) > 0:
            logger.debug("STDOUT: {}".format(stdout_bytes.decode("utf-8").strip()))
        if stderr_bytes is not None and len(stderr_bytes) > 0:
            logger.debug("STDERR: {}".format(stderr_bytes.decode("utf-8").strip()))

    return returncode


def _run_subprocess(
        command: str,
        args: List[Any],
        cwd: Path = None,
        shell: bool = False,
        environment: Optional[Dict[str, str]] = None,
        stdout: Union[int, TextIOBase, TextIO] = None,
        stdin: int = subprocess.PIPE,
        stderr: int = subprocess.PIPE,
        piped_command: Optional[str] = None
) -> Tuple[bytes, bytes, int]:
    try:
        if piped_command is None:
            result = subprocess.run(
                [command] + args,
                stdin=stdin,
                stderr=stderr,
                stdout=stdout,
                shell=shell,
                cwd=None if cwd is None else str(cwd),
                env=environment
            )
            return result.stdout, result.stderr, result.returncode
        else:
            if len(args) > 0:
                cmd_str = '{} {} | {}'.format(command, ' '.join(args), piped_command)
            else:
                cmd_str = '{} | {}'.format(command, piped_command)
            p = subprocess.Popen(
                cmd_str,
                stdin=stdin,
                stderr=stderr,
                stdout=stdout,
                shell=shell,
                cwd=None if cwd is None else str(cwd),
                env=environment
            )
            stdout, stderr = p.communicate()
            return stdout, stderr, p.returncode
    except FileNotFoundError as e:
        raise RuntimeError(f"Encountered file error running subprocess. Is `{command}` installed?") from e
