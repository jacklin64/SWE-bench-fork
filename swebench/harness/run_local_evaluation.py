from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path, PurePosixPath

if platform.system() == "Linux":
    import resource

import shlex

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.docker_build import (
    BuildImageError,
    close_logger,
    setup_logger,
)
from swebench.harness.eval import get_log_dir
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.harness.modal_eval import (
    run_instances_modal,
    validate_modal_credentials,
)
from swebench.harness.tracto_eval import (
    get_tracto_eval_run_dir,
    run_instances_tracto,
    validate_tracto_env_vars,
)
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import (
    EvaluationError,
    get_predictions_from_file,
    load_swebench_dataset,
    run_threadpool,
    str2bool,
)

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def _coerce_json_object_field(
    value: object,
    *,
    field_name: str,
    instance_id: str,
) -> dict | None:
    """Parse JSON string fields (e.g. NeMo-Gym ``swe_rebench.jsonl``) into dicts for ``make_test_spec``."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON for {field_name!r} on instance {instance_id!r}: {e}"
            ) from e
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Expected JSON object for {field_name!r} on instance {instance_id!r}, "
                f"got {type(parsed).__name__}"
            )
        return parsed
    raise TypeError(
        f"Expected dict or JSON str for {field_name!r} on instance {instance_id!r}, "
        f"got {type(value).__name__}"
    )


def _normalize_dataset_instance_row(instance: object) -> object:
    """Copy a dataset row and decode string ``install_config`` / ``meta`` when present."""
    if not isinstance(instance, dict):
        return instance
    iid = str(instance.get(KEY_INSTANCE_ID, "<unknown>"))
    out = dict(instance)
    if "install_config" in out:
        ic = out["install_config"]
        if isinstance(ic, str):
            parsed = _coerce_json_object_field(
                ic, field_name="install_config", instance_id=iid
            )
            if parsed is None:
                del out["install_config"]
            else:
                out["install_config"] = parsed
    if "meta" in out:
        m = out["meta"]
        if isinstance(m, str):
            parsed = _coerce_json_object_field(m, field_name="meta", instance_id=iid)
            if parsed is None:
                del out["meta"]
            else:
                out["meta"] = parsed
    return out


# Flags consumed by the host-side Apptainer wrapper (not forwarded to the inner harness).
_APPTAINER_ONE_ARG = frozenset(
    {
        "--apptainer-sif",
        "--apptainer-exec",
        "--apptainer-python",
        "--apptainer-bind",
    }
)
_APPTAINER_TOGGLE = frozenset({"--apptainer-nv"})


def _resolve_apptainer_exec(cmd: str) -> str:
    """Resolve ``apptainer`` / ``singularity`` to an absolute path for ``subprocess`` (non-login PATH)."""
    c = (cmd or "apptainer").strip() or "apptainer"
    if os.path.isabs(c):
        p = Path(c)
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"Apptainer executable not found: {c}")
    resolved = shutil.which(c)
    if resolved:
        return resolved
    if c == "singularity":
        fallback = shutil.which("apptainer")
        if fallback:
            return fallback
    raise FileNotFoundError(
        f"Container executable not found: {c!r} (not on PATH). "
        "Install Apptainer, export PATH (e.g. /usr/bin), or pass --apptainer-exec /full/path/to/apptainer"
    )


def _strip_apptainer_flags_from_argv(argv: list[str]) -> list[str]:
    """Return argv with Apptainer-only options removed (for inner ``python -m ...``)."""
    out: list[str] = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a in _APPTAINER_TOGGLE:
            i += 1
            continue
        if a.startswith("--apptainer-bind="):
            i += 1
            continue
        if a in _APPTAINER_ONE_ARG:
            if i + 1 >= n:
                raise SystemExit(f"{a} requires a value")
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def _reexec_under_apptainer(
    *,
    sif: str,
    apptainer_exec: str,
    apptainer_python: str,
    apptainer_bind: list[str] | None,
    apptainer_nv: bool,
) -> int:
    """Re-run this module inside ``apptainer exec`` (same style as nemo-skills ``swebench.py``).

    Mirrors the host command pattern::

        apptainer exec --writable-tmpfs --cleanenv --no-mount home,tmp,bind-paths \\
          [--mount type=bind,src=...,dst=... ...] \\
          <sif> bash -lc '<python> -m swebench.harness.run_local_evaluation <args>'

    Pass ``--apptainer-bind SRC:DST`` (repeatable) for any paths the inner run needs
    (e.g. ``/nemo_run/code:/nemo_run/code``, ``/root:/root_mount,ro``, predictions dirs).
    """
    forward = _strip_apptainer_flags_from_argv(sys.argv[1:])

    inner_parts = [
        '/root/SWE-bench/venv/bin/python',
        "-m",
        "swebench.harness.run_local_evaluation",
        *forward,
    ]
    inner = " ".join(shlex.quote(p) for p in inner_parts)

    exe = _resolve_apptainer_exec(apptainer_exec)
    cmd: list[str] = [
        exe,
        "exec",
        "--writable-tmpfs",
        "--cleanenv",
        "--no-mount",
        "home,tmp,bind-paths",
    ]
    if apptainer_nv:
        cmd.append("--nv")
    for b in apptainer_bind or []:
        src, sep, dst = b.partition(":")
        if not sep:
            raise SystemExit(
                f"Invalid --apptainer-bind {b!r}; expected SRC:DST (optional ,ro on dst)"
            )
        cmd.extend(["--mount", f"type=bind,src={src},dst={dst}"])
    cmd.extend([sif, "bash", "-lc", inner])

    print("Re-exec under Apptainer:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return int(subprocess.call(cmd))


def exec_run_with_timeout(cmd: str, timeout: int | None = 60) -> tuple[str, bool, float]:
    """Run a shell command locally with a timeout (no Docker)."""
    exec_result = b""
    process: subprocess.Popen | None = None
    exception: BaseException | None = None
    timed_out = False

    def run_command() -> None:
        nonlocal exec_result, process, exception
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
            )
            exec_result, _ = process.communicate()
        except Exception as e:
            exception = e

    thread = threading.Thread(target=run_command)
    start_time = time.time()
    thread.start()
    thread.join(timeout if timeout is not None else 60**4)

    if exception:
        raise exception

    if thread.is_alive():
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        timed_out = True
    end_time = time.time()
    return exec_result.decode(errors="replace"), timed_out, end_time - start_time


def run_instance(
    test_spec: TestSpec,
    pred: dict,
    rm_image: bool,
    force_rebuild: bool,
    client: object,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False,
) -> dict:
    """
    Run a single instance on the current machine (e.g. already inside Apptainer).

    Same pattern as
    https://github.com/Kipok/SWE-bench/blob/main/swebench/harness/run_local_evaluation.py
    — subprocess + ``DOCKER_WORKDIR`` / ``DOCKER_PATCH``, no ``docker run`` / no container API.
    ``client`` is ignored (kept for call compatibility with the old threadpool payloads).
    """
    del rm_image, force_rebuild, client

    instance_id = test_spec.instance_id
    log_dir = get_log_dir(pred, run_id, instance_id)

    report_path = log_dir / LOG_REPORT
    if rewrite_reports:
        test_output_path = log_dir / LOG_TEST_OUTPUT
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return {
            "completed": True,
            "resolved": report[instance_id]["resolved"],
        }
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {
            "completed": True,
            "resolved": report[instance_id]["resolved"],
        }

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    eval_completed = False
    report: dict = {}
    try:
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}..."
        )
        shutil.copy2(patch_file, PurePosixPath(DOCKER_PATCH))

        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = subprocess.run(
                f"{git_apply_cmd} {DOCKER_PATCH}",
                cwd=DOCKER_WORKDIR,
                shell=True,
                capture_output=True,
            )
            if val.returncode == 0:
                check_orig = subprocess.run(
                    f"find {DOCKER_WORKDIR} -name '*.orig' -type f | head -1",
                    cwd=DOCKER_WORKDIR,
                    shell=True,
                    capture_output=True,
                )
                if check_orig.stdout.decode(UTF8).strip():
                    restore_val = subprocess.run(
                        f"find {DOCKER_WORKDIR} -name '*.orig' -exec sh -c 'mv \"$1\" \"${{1%.orig}}\"' _ {{}} \\;",
                        cwd=DOCKER_WORKDIR,
                        shell=True,
                        capture_output=True,
                    )
                    if restore_val.returncode == 0:
                        logger.info(
                            "Detected reverse patch, restored files from .orig and treating as already applied"
                        )
                        logger.info(f"{APPLY_PATCH_PASS}:\n{val.stdout.decode(UTF8)}")
                        applied_patch = True
                        break
                else:
                    logger.info(f"{APPLY_PATCH_PASS}:\n{val.stdout.decode(UTF8)}")
                    applied_patch = True
                    break
            else:
                logger.info(f"Failed to apply patch to container: {git_apply_cmd}")
        if not applied_patch:
            err_out = (val.stderr or val.stdout or b"").decode(UTF8)
            logger.info(f"{APPLY_PATCH_FAIL}:\n{err_out}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{err_out}",
                logger,
            )

        git_diff_output_before = (
            subprocess.run(
                "git -c core.fileMode=false diff",
                cwd=DOCKER_WORKDIR,
                shell=True,
                capture_output=True,
            )
            .stdout.decode(UTF8)
            .strip()
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(f"Eval script for {instance_id} written to {eval_file}...")
        shutil.copy2(eval_file, PurePosixPath("/eval.sh"))

        test_output, timed_out, total_runtime = exec_run_with_timeout(
            "/bin/bash /eval.sh", timeout
        )
        test_output_path = log_dir / LOG_TEST_OUTPUT
        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )

        git_diff_output_after = (
            subprocess.run(
                "git -c core.fileMode=false diff",
                cwd=DOCKER_WORKDIR,
                shell=True,
                capture_output=True,
            )
            .stdout.decode(UTF8)
            .strip()
        )
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info("Git diff changed after running eval script")

        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        eval_completed = True
    except (EvaluationError, BuildImageError) as e:
        logger.info(traceback.format_exc())
        print(e)
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
    finally:
        close_logger(logger)
    return {
        "completed": eval_completed,
        "resolved": report.get(instance_id, {}).get("resolved", False),
    }


def run_instances(
    predictions: dict,
    instances: list,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    max_workers: int,
    run_id: str,
    timeout: int,
    namespace: str | None = "swebench",
    instance_image_tag: str = "latest",
    rewrite_reports: bool = False,
):
    """
    Run all instances for the given predictions in parallel (no Docker on host).

    ``cache_level``, ``clean``, and image tags are accepted for CLI compatibility; local
    evaluation assumes the environment is already the target image (e.g. Apptainer).

    Rows from JSONL exports (e.g. ``swe_rebench.jsonl``) may store ``install_config`` and
    ``meta`` as JSON strings; they are parsed to dicts before ``make_test_spec``.
    """
    del cache_level, clean
    normalized = [_normalize_dataset_instance_row(i) for i in instances]
    test_specs = list(
        map(
            lambda instance: make_test_spec(
                instance,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
            ),
            normalized,
        )
    )

    payloads = []
    for test_spec in test_specs:
        payloads.append(
            (
                test_spec,
                predictions[test_spec.instance_id],
                False,
                force_rebuild,
                None,
                run_id,
                timeout,
                rewrite_reports,
            )
        )

    print(f"Running {len(instances)} instances...")
    run_threadpool(run_instance, payloads, max_workers)
    print("All instances run.")


def get_dataset_from_preds(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions: dict,
    run_id: str,
    rewrite_reports: bool,
    exclude_completed: bool = True,
):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # load dataset
    dataset = load_swebench_dataset(dataset_name, split)
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

    if instance_ids:
        # check that all instance IDs have predictions
        missing_preds = set(instance_ids) - set(predictions.keys())
        if missing_preds:
            print(
                f"Warning: Missing predictions for {len(missing_preds)} instance IDs."
            )

    # check that all prediction IDs are in the dataset
    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        raise ValueError(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )
    if instance_ids:
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]

    if rewrite_reports:
        # we only return instances that have existing test outputs
        test_output_ids = set()
        for instance in dataset:
            if instance[KEY_INSTANCE_ID] not in predictions:
                continue
            prediction = predictions[instance[KEY_INSTANCE_ID]]
            test_output_file = (
                RUN_EVALUATION_LOG_DIR
                / run_id
                / prediction["model_name_or_path"].replace("/", "__")
                / prediction[KEY_INSTANCE_ID]
                / "test_output.txt"
            )
            if test_output_file.exists():
                test_output_ids.add(instance[KEY_INSTANCE_ID])
        dataset = [
            i
            for i in dataset
            if i[KEY_INSTANCE_ID] in prediction_ids
            and i[KEY_INSTANCE_ID] in test_output_ids
        ]
        return dataset

    # check which instance IDs have already been run
    completed_ids = set()
    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            # skip instances without predictions
            continue
        prediction = predictions[instance[KEY_INSTANCE_ID]]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction[KEY_MODEL].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / LOG_REPORT
        )
        if report_file.exists():
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        # filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]

    empty_patch_ids = {
        k
        for k, v in predictions.items()
        if v[KEY_PREDICTION] == "" or v[KEY_PREDICTION] is None
    }

    # filter dataset to only instances with predictions
    dataset = [
        i
        for i in dataset
        if i[KEY_INSTANCE_ID] in prediction_ids
        and i[KEY_INSTANCE_ID] not in empty_patch_ids
    ]
    return dataset


def main(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions_path: str,
    max_workers: int,
    force_rebuild: bool,
    cache_level: str,
    clean: bool,
    open_file_limit: int,
    run_id: str,
    timeout: int,
    namespace: str | None,
    rewrite_reports: bool,
    modal: bool,
    tracto: bool,
    instance_image_tag: str = "latest",
    report_dir: str = ".",
):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    namespace = None if namespace == "" else namespace

    if dataset_name == "SWE-bench/SWE-bench_Multimodal" and split == "test":
        print(
            "⚠️ Local evaluation for the test split of SWE-bench Multimodal is not supported. "
            "Please check out sb-cli (https://github.com/swe-bench/sb-cli/) for instructions on how to submit predictions."
        )
        return

    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    # TODO: validate run_id as it's used in container name
    if report_dir is not None:
        report_dir = Path(report_dir)
        if not report_dir.exists():
            report_dir.mkdir(parents=True)

    if force_rebuild and namespace is not None:
        raise ValueError("Cannot force rebuild and use a namespace at the same time.")

    # load predictions as map of instance_id to prediction
    predictions = get_predictions_from_file(predictions_path, dataset_name, split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    dataset = get_dataset_from_preds(
        dataset_name, split, instance_ids, predictions, run_id, rewrite_reports
    )
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids)

    if modal and tracto:
        raise ValueError("Only one of --modal and --tracto can be used.")

    if modal:
        # run instances on Modal
        if not dataset:
            print("No instances to run.")
        else:
            validate_modal_credentials()
            run_instances_modal(predictions, dataset, full_dataset, run_id, timeout)
        return

    if tracto:
        # run instances on Tracto
        if not dataset:
            print("No instances to run.")
        else:
            validate_tracto_env_vars()

            run_instances_tracto(
                predictions,
                dataset,
                full_dataset,
                run_id,
                timeout,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                tracto_run_dir=get_tracto_eval_run_dir(run_id),
            )
        return

    # Local: no Docker — assume already in instance env (e.g. Apptainer .sif).
    if platform.system() == "Linux":
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    if not dataset:
        print("No instances to run.")
    else:
        run_instances(
            predictions,
            dataset,
            cache_level,
            clean,
            force_rebuild,
            max_workers,
            run_id,
            timeout,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            rewrite_reports=rewrite_reports,
        )

    return make_run_report(predictions, full_dataset, run_id, None)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run evaluation harness for the given dataset and predictions.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Common args
    parser.add_argument(
        "--dataset_name",
        default="nebius/SWE-rebench",
        type=str,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split of the dataset"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        help="Path to predictions file - if 'gold', uses gold predictions",
        required=True,
    )

    # Local execution args
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of workers (should be <= 75%% of CPU cores)",
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout (in seconds) for running tests for each instance",
    )
    parser.add_argument(
        "--force_rebuild",
        type=str2bool,
        default=False,
        help="Force rebuild of all images",
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )
    parser.add_argument(
        "--namespace", type=str, default="swebench", help="Namespace for images"
    )
    parser.add_argument(
        "--instance_image_tag", type=str, default="latest", help="Instance image tag"
    )
    parser.add_argument(
        "--rewrite_reports",
        type=str2bool,
        default=False,
        help="Doesn't run new instances, only writes reports for instances with existing test outputs",
    )
    parser.add_argument(
        "--report_dir", type=str, default=".", help="Directory to write reports to"
    )

    # Modal execution args
    parser.add_argument("--modal", type=str2bool, default=False, help="Run on Modal")

    # Traco execution args
    parser.add_argument("--tracto", type=str2bool, default=False, help="Run on Tracto")

    # Apptainer: re-exec this module inside a .sif (same idea as nemo-skills swebench.py
    # wrapping ``python -m swebench.harness.run_local_evaluation`` in ``apptainer exec``).
    parser.add_argument(
        "--apptainer-sif",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "If set, do not run Docker on the host: instead ``apptainer exec`` into this "
            "image and run ``<apptainer-python> -m swebench.harness.run_local_evaluation`` "
            "with all other CLI args forwarded. Add ``--apptainer-bind`` for each host path "
            "the inner run must see."
        ),
    )
    parser.add_argument(
        "--apptainer-exec",
        type=str,
        default="apptainer",
        help="Apptainer/Singularity executable name or path (default: apptainer).",
    )
    parser.add_argument(
        "--apptainer-python",
        type=str,
        default="/usr/bin/python3",
        help=(
            "Python *inside the .sif* for re-exec (not the host venv). "
            "SWE-rebench instance images typically use ``/usr/bin/python3``; "
            "classic SWE Docker-style images may need ``/root/SWE-bench/venv/bin/python``."
        ),
    )
    parser.add_argument(
        "--apptainer-bind",
        action="append",
        default=None,
        metavar="SRC:DST",
        help=(
            "Extra bind mount ``SRC:DST`` (repeatable). Dst may include ``,ro`` suffix "
            "as in Apptainer mount syntax."
        ),
    )
    parser.add_argument(
        "--apptainer-nv",
        action="store_true",
        help="Pass ``--nv`` to ``apptainer exec`` (GPU).",
    )

    args = parser.parse_args()
    ns = vars(args)
    if ns.get("apptainer_sif"):
        rc = _reexec_under_apptainer(
            sif=ns["apptainer_sif"],
            apptainer_exec=ns["apptainer_exec"],
            apptainer_python=ns["apptainer_python"],
            apptainer_bind=list(ns.get("apptainer_bind") or []),
            apptainer_nv=bool(ns.get("apptainer_nv")),
        )
        raise SystemExit(rc)
    for k in (
        "apptainer_sif",
        "apptainer_exec",
        "apptainer_python",
        "apptainer_bind",
        "apptainer_nv",
    ):
        ns.pop(k, None)
    main(**ns)
