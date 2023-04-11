from pathlib import Path
import subprocess
from typing import List, Optional, Dict, Any, Union
from io import TextIOBase

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class CommandLineException(Exception):
    def __init__(self, cmd, exit_code):
        super().__init__("`{}` encountered an error. (code: {})".format(cmd, exit_code))
        self.cmd = cmd
        self.exit_code = exit_code


def call_command(command: str,
                 args: List[Any],
                 cwd: Path = None,
                 shell: bool = False,
                 environment: Optional[Dict[str, str]] = None,
                 stdout: Union[None, Path, TextIOBase] = None,
                 stdin: int = subprocess.PIPE,
                 stderr: int = subprocess.PIPE,
                 silent: bool = False) -> int:
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
        logger.debug("EXECUTE {cwdstr}{cmd} {arguments}".format(
            cmd=command,
            cwdstr="" if cwd is None else "[cwd={}] ".format(cwd),
            arguments=" ".join(args)
        ))

    if stdout is None:
        p = _run_subprocess(command, args, cwd, shell, environment, subprocess.PIPE, stdin, stderr)
    elif isinstance(stdout, Path):
        if not silent:
            logger.debug("STDOUT redirect to {}.".format(stdout))
        with open(stdout, 'wt') as f:
            p = _run_subprocess(command, args, cwd, shell, environment, f, stdin, stderr)
    elif isinstance(stdout, TextIOBase):
        # Some other arbitrary textio (which might not have a fileno)
        p = _run_subprocess(command, args, cwd, shell, environment, subprocess.PIPE, stdin, stderr)
        stdout.write(p.stdout.decode('utf-8'))
    else:
        raise RuntimeError("Unsupported `stdout` argument type: {}".format(type(stdout)))

    stdout_bytes = p.stdout
    stderr_bytes = p.stderr

    if not silent:
        if stdout_bytes is not None and len(stdout_bytes) > 0:
            logger.debug("STDOUT: {}".format(stdout_bytes.decode("utf-8").strip()))
        if stderr_bytes is not None and len(stderr_bytes) > 0:
            logger.debug("STDERR: {}".format(stderr_bytes.decode("utf-8").strip()))

    return p.returncode


def _run_subprocess(
        command: str,
        args: List[Any],
        cwd: Path = None,
        shell: bool = False,
        environment: Optional[Dict[str, str]] = None,
        stdout: Union[int, TextIOBase] = None,
        stdin: int = subprocess.PIPE,
        stderr: int = subprocess.PIPE,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [command] + args,
            stdin=stdin,
            stderr=stderr,
            stdout=stdout,
            shell=shell,
            cwd=None if cwd is None else str(cwd),
            env=environment
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Encountered file error running subprocess. Is `{command}` installed?") from e
