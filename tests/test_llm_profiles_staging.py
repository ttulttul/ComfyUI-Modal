"""Tests for the llm profiles staging boundary."""

from __future__ import annotations

from modal_llm_test_support import *  # noqa: F401,F403

def test_curated_profile_is_revision_pinned_and_found_in_nested_payload(
    llm_profiles_module: Any,
) -> None:
    """The registry should reject drift across split-payload discovery."""
    profiles = llm_profiles_module.load_llm_profiles()
    profile = profiles["smolvlm2-2.2b-instruct"]

    assert profile.revision == "482adb537c021c86670beed01cd58990d01e72e4"
    assert profile.modalities == frozenset({"text", "image", "video", "file"})
    assert profile.trust_remote_code is False
    payload = {
        "split_proxy_payloads": [
            {
                "subgraph_prompt": {
                    "12": {
                        "class_type": llm_profiles_module.MODAL_LLM_NODE_ID,
                        "inputs": {"model_profile": profile.profile_id},
                    }
                }
            }
        ]
    }

    assert llm_profiles_module.llm_profile_ids_from_payload(payload) == (
        profile.profile_id,
    )

def test_profile_registry_rejects_mutable_revision(
    llm_profiles_module: Any,
) -> None:
    """Curated profiles must use an immutable commit rather than main."""
    with pytest.raises(ValueError, match="exact 40-character"):
        llm_profiles_module.LLMModelProfile.from_mapping(
            {
                "id": "unsafe",
                "repository": "owner/model",
                "revision": "main",
                "dtype": "bfloat16",
                "modalities": ["text"],
                "estimated_vram_gb": 1,
                "max_context_tokens": 1024,
                "max_images": 1,
                "max_video_frames": 1,
                "max_file_bytes": 1,
                "max_file_characters": 1,
            }
        )

@pytest.mark.parametrize(
    ("architecture", "quantization_config", "backend", "quantization_method"),
    [
        (
            "Qwen3_5ForConditionalGeneration",
            {"quant_method": "fp8", "fmt": "e4m3"},
            "vllm",
            "fp8",
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            {"quant_method": "modelopt", "quant_algo": "NVFP4"},
            "vllm",
            "modelopt_fp4",
        ),
        (
            "MuseGlimmerForConditionalGeneration",
            {},
            "transformers",
            "none",
        ),
    ],
)
def test_compatibility_policy_selects_requested_model_backends(
    llm_compatibility_module: Any,
    architecture: str,
    quantization_config: dict[str, str],
    backend: str,
    quantization_method: str,
) -> None:
    """The three live-canary model families should resolve before weight download."""
    decision = llm_compatibility_module.resolve_compatibility(
        {
            "architectures": [architecture],
            "dtype": "bfloat16",
            "text_config": {"max_position_embeddings": 262144},
            "quantization_config": quantization_config,
        },
        artifact_bytes=32 * 1024**3,
    )

    assert decision.backend == backend
    assert decision.quantization_method == quantization_method
    assert decision.default_context_tokens == 32768
    assert decision.advertised_context_tokens == 262144
    assert decision.reasoning_parser == ("qwen3" if backend == "vllm" else "none")

@pytest.mark.parametrize(
    ("architecture", "quantization", "reasoning_parser"),
    [
        ("SmolVLMForConditionalGeneration", {}, "none"),
        ("MuseGlimmerForConditionalGeneration", {}, "none"),
        (
            "Qwen3_5ForConditionalGeneration",
            {"bits": 4, "group_size": 64, "mode": "affine"},
            "qwen3",
        ),
    ],
)
def test_apple_local_compatibility_selects_mlx_vlm(
    llm_compatibility_module: Any,
    architecture: str,
    quantization: dict[str, Any],
    reasoning_parser: str,
) -> None:
    """Reviewed Apple-local architectures should resolve to the MLX adapter."""
    decision = llm_compatibility_module.resolve_compatibility(
        {
            "architectures": [architecture],
            "dtype": "bfloat16",
            "text_config": {"max_position_embeddings": 65536},
            "quantization": quantization,
        },
        artifact_bytes=2 * 1024**3,
        execution_target="local_apple",
    )

    assert decision.backend == "mlx_vlm"
    assert decision.reasoning_parser == reasoning_parser
    assert decision.runtime_requirements == ("mlx-vlm==0.6.15",)
    assert decision.estimated_vram_gb < 5

def test_apple_local_compatibility_rejects_cuda_quantization(
    llm_compatibility_module: Any,
) -> None:
    """Apple resolution should fail before downloading CUDA-specific weights."""
    with pytest.raises(ValueError, match="not an MLX checkpoint format"):
        llm_compatibility_module.resolve_compatibility(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {"max_position_embeddings": 32768},
                "quantization_config": {"quant_method": "fp8"},
            },
            artifact_bytes=1024,
            execution_target="local_apple",
        )

def test_curated_profile_adapts_to_apple_without_mutating_modal_profile(
    llm_profiles_module: Any,
) -> None:
    """One saved curated id should resolve independently for both targets."""
    modal_profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")

    local_profile = llm_profiles_module.profile_for_execution_target(
        modal_profile,
        "local_apple",
    )

    assert modal_profile.backend == "transformers"
    assert modal_profile.execution_target == "modal"
    assert local_profile.profile_id == modal_profile.profile_id
    assert local_profile.repository == modal_profile.repository
    assert local_profile.backend == "mlx_vlm"
    assert local_profile.execution_target == "local_apple"

def test_generated_profile_requires_matching_content_digest(
    llm_profiles_module: Any,
) -> None:
    """A generated manifest must not be mutable under a stable profile identifier."""
    digest = "a" * 64
    with pytest.raises(ValueError, match="does not match its content digest"):
        llm_profiles_module.LLMModelProfile.from_mapping(
            {
                "id": "hf-" + "b" * 64,
                "repository": "owner/model",
                "revision": "1" * 40,
                "dtype": "bfloat16",
                "modalities": ["text"],
                "estimated_vram_gb": 10,
                "max_context_tokens": 1024,
                "max_images": 1,
                "max_video_frames": 1,
                "max_file_bytes": 1,
                "max_file_characters": 1,
                "schema_version": 2,
                "source": "generated",
                "profile_digest": digest,
                "backend": "transformers",
            }
        )

def test_cpu_resolver_pins_and_persists_generated_profile(
    llm_resolver_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """One model ID should become a stable manifest without downloading weights."""
    revision = "9" * 40
    config_path = tmp_path / "downloaded-config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "dtype": "bfloat16",
                "text_config": {"max_position_embeddings": 262144},
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                },
            }
        ),
        encoding="utf-8",
    )
    model_info = {
        "id": "owner/model",
        "sha": revision,
        "siblings": [
            {"rfilename": "config.json", "size": 1000},
            {"rfilename": "model-00001.safetensors", "size": 20 * 1024**3},
            {"rfilename": "model-00002.safetensors", "size": 10 * 1024**3},
        ],
        "securityStatus": {"scansDone": True, "filesWithIssues": []},
    }

    class FakeApi:
        """Return immutable test metadata from model_info."""

        def model_info(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
            """Validate the metadata-only resolver request."""
            assert repo_id == "owner/model"
            assert kwargs["revision"] == "release"
            assert kwargs["files_metadata"] is True
            return model_info

    download_calls: list[dict[str, Any]] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        """Return only the config file during resolution."""
        download_calls.append(kwargs)
        return str(config_path)

    first = llm_resolver_module.resolve_model_profile(
        "owner/model@release",
        tmp_path,
        api=FakeApi(),
        hf_hub_download=fake_hf_hub_download,
    )
    second = llm_resolver_module.resolve_model_profile(
        "owner/model@release",
        tmp_path,
        api=FakeApi(),
        hf_hub_download=fake_hf_hub_download,
    )

    assert first.profile.profile_id == second.profile.profile_id
    assert first.profile.revision == revision
    assert first.profile.backend == "vllm"
    assert first.profile.reasoning_parser == "qwen3"
    assert first.profile.quantization_method == "modelopt_fp4"
    assert first.profile.artifact_bytes == 30 * 1024**3
    assert first.manifest_created is True
    assert second.manifest_created is False
    assert len(download_calls) == 2
    loaded = llm_profiles_module.get_llm_profile(
        first.profile.profile_id,
        storage_root=tmp_path,
    )
    assert loaded == first.profile

def test_local_resolver_creates_target_specific_mlx_profile(
    llm_resolver_module: Any,
    tmp_path: Path,
) -> None:
    """The same user-facing model reference should get an Apple-local manifest."""
    config_path = tmp_path / "mlx-config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "dtype": "bfloat16",
                "text_config": {"max_position_embeddings": 65536},
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeApi:
        """Return one immutable MLX-format repository."""

        def model_info(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
            """Return compatible metadata without downloading model weights."""
            del repo_id, kwargs
            return {
                "sha": "7" * 40,
                "siblings": [
                    {"rfilename": "config.json", "size": 1000},
                    {"rfilename": "model.safetensors", "size": 2 * 1024**3},
                ],
                "securityStatus": {"scansDone": True, "filesWithIssues": []},
            }

    result = llm_resolver_module.resolve_model_profile(
        "mlx-community/Qwen3.5-2B-4bit",
        tmp_path,
        api=FakeApi(),
        hf_hub_download=lambda **kwargs: str(config_path),
        execution_target="local_apple",
    )

    assert result.profile.backend == "mlx_vlm"
    assert result.profile.execution_target == "local_apple"
    assert result.profile.quantization_method == "mlx_affine_4bit"
    assert result.profile.runtime_requirements == ("mlx-vlm==0.6.15",)

def test_cpu_resolver_rejects_unknown_architecture_before_weights(
    llm_resolver_module: Any,
    tmp_path: Path,
) -> None:
    """Compatibility errors should happen after config-only inspection."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["UnreviewedForConditionalGeneration"],
                "max_position_embeddings": 4096,
            }
        ),
        encoding="utf-8",
    )

    class FakeApi:
        """Return a safe but unsupported test repository."""

        def model_info(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
            """Return metadata without performing network access."""
            del repo_id, kwargs
            return {
                "sha": "8" * 40,
                "siblings": [
                    {"rfilename": "config.json", "size": 1},
                    {"rfilename": "model.safetensors", "size": 1024},
                ],
                "securityStatus": {"scansDone": True},
            }

    with pytest.raises(ValueError, match="not supported"):
        llm_resolver_module.resolve_model_profile(
            "owner/unknown",
            tmp_path,
            api=FakeApi(),
            hf_hub_download=lambda **kwargs: str(config_path),
        )
    assert not (tmp_path / "llm_models").exists()

def test_cpu_resolver_wraps_gated_config_download_error(
    llm_resolver_module: Any,
    tmp_path: Path,
) -> None:
    """A gated config failure should cross Modal as a plain actionable ValueError."""

    class FakeApi:
        """Return metadata visible without gated file access."""

        def model_info(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
            """Return one otherwise compatible repository."""
            del repo_id, kwargs
            return {
                "sha": "8" * 40,
                "siblings": [
                    {"rfilename": "config.json", "size": 1},
                    {"rfilename": "model.safetensors", "size": 1024},
                ],
                "securityStatus": {"scansDone": True},
            }

    def denied_download(**kwargs: Any) -> str:
        """Simulate gated file access after public metadata resolution."""
        del kwargs
        raise OSError("401 Unauthorized")

    with pytest.raises(
        ValueError,
        match="Unable to download config.json.*HF_TOKEN",
    ):
        llm_resolver_module.resolve_model_profile(
            "owner/gated",
            tmp_path,
            api=FakeApi(),
            hf_hub_download=denied_download,
        )

def test_local_resolver_reports_local_hugging_face_token_location(
    llm_resolver_module: Any,
    tmp_path: Path,
) -> None:
    """Gated local models should not direct users to a Modal secret."""

    class DeniedApi:
        """Reject model metadata as a gated repository would."""

        def model_info(self, repo_id: str, **kwargs: Any) -> Any:
            """Raise a deterministic authorization error."""
            del repo_id, kwargs
            raise OSError("401 Unauthorized")

    with pytest.raises(ValueError, match="set HF_TOKEN in the local ComfyUI"):
        llm_resolver_module.resolve_model_profile(
            "owner/gated",
            tmp_path,
            api=DeniedApi(),
            hf_hub_download=lambda **kwargs: "unused",
            execution_target="local_apple",
        )

def test_cpu_stager_writes_completion_marker_and_reuses_snapshot(
    llm_staging_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A completed immutable snapshot should not download twice."""
    monkeypatch.setenv("HF_TOKEN", "test-hugging-face-token")
    calls: list[dict[str, Any]] = []
    progress: list[Any] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        """Materialize the minimum expected Hugging Face snapshot."""
        calls.append(kwargs)
        snapshot_path = Path(kwargs["local_dir"])
        snapshot_path.mkdir(parents=True, exist_ok=True)
        (snapshot_path / "config.json").write_text("{}", encoding="utf-8")
        progress_bar = kwargs["tqdm_class"](
            total=2,
            unit="files",
            file=StringIO(),
        )
        progress_bar.update(1)
        progress_bar.update(1)
        progress_bar.close()
        return str(snapshot_path)

    first = llm_staging_module.stage_model_profile(
        "smolvlm2-2.2b-instruct",
        tmp_path,
        snapshot_download=fake_snapshot_download,
        progress_callback=progress.append,
    )
    second = llm_staging_module.stage_model_profile(
        "smolvlm2-2.2b-instruct",
        tmp_path,
        snapshot_download=fake_snapshot_download,
    )

    assert first.downloaded is True
    assert second.downloaded is False
    assert len(calls) == 1
    assert calls[0]["revision"] == "482adb537c021c86670beed01cd58990d01e72e4"
    assert calls[0]["token"] == "test-hugging-face-token"
    assert "*.safetensors" in calls[0]["allow_patterns"]
    assert "*.bin" not in calls[0]["allow_patterns"]
    assert calls[0]["tqdm_class"].__name__ == "SnapshotProgressTqdm"
    assert [event.value for event in progress if event.stage == "download"] == [
        0.0,
        1.0,
        2.0,
    ]
    assert [event.stage for event in progress] == [
        "snapshot_check",
        "disk_check",
        "download_preparing",
        "download",
        "download",
        "download",
        "staged",
    ]
    assert {event.model_reference for event in progress} == {
        "smolvlm2-2.2b-instruct"
    }
    assert llm_staging_module.is_model_snapshot_staged(
        tmp_path,
        llm_staging_module.get_llm_profile("smolvlm2-2.2b-instruct"),
    )

def test_gguf_stager_downloads_only_selected_model_and_pinned_tokenizer(
    llm_staging_module: Any,
    tmp_path: Path,
) -> None:
    """GGUF staging must not fetch every quant or the tokenizer repo's weights."""
    profile = llm_staging_module.get_llm_profile(
        "huihui-qwen3.8-27b-abliterated-q2-k-gguf"
    )
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        """Materialize the selected GGUF and tokenizer sentinel."""
        calls.append(kwargs)
        snapshot_path = Path(kwargs["local_dir"])
        snapshot_path.mkdir(parents=True, exist_ok=True)
        if kwargs["repo_id"] == profile.repository:
            (snapshot_path / profile.backend_option("model_filename")).write_bytes(
                b"gguf"
            )
            (snapshot_path / profile.backend_option("mmproj_filename")).write_bytes(
                b"mmproj"
            )
        else:
            (snapshot_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return str(snapshot_path)

    result = llm_staging_module.stage_model_profile(
        profile.profile_id,
        tmp_path,
        profile=profile,
        snapshot_download=fake_snapshot_download,
    )

    assert result.downloaded is True
    assert len(calls) == 2
    assert calls[0]["allow_patterns"][-2:] == (
        profile.backend_option("model_filename"),
        profile.backend_option("mmproj_filename"),
    )
    assert "*.gguf" not in calls[0]["allow_patterns"]
    assert "*.safetensors" not in calls[1]["allow_patterns"]
    assert calls[1]["repo_id"] == profile.backend_option("tokenizer_repository")
    assert calls[1]["revision"] == profile.backend_option("tokenizer_revision")
    assert llm_staging_module.is_model_snapshot_staged(tmp_path, profile)

def test_provider_neutral_stager_resolves_and_stages_model_reference(
    llm_staging_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Modal and SSH staging should share one immutable resolution implementation."""
    profile = SimpleNamespace(
        profile_id="hf-" + "c" * 64,
        repository="owner/model",
        revision="9" * 40,
        backend="vllm",
        quantization_method="compressed-tensors",
        artifact_bytes=12 * 1024**3,
    )
    staged = llm_staging_module.StagedModelSnapshot(
        profile_id=profile.profile_id,
        repository=profile.repository,
        revision=profile.revision,
        path=str(tmp_path / "snapshot"),
        downloaded=True,
        elapsed_seconds=3.0,
    )
    progress: list[Any] = []
    monkeypatch.setattr(
        llm_staging_module,
        "_resolve_profile_for_staging",
        lambda _reference, _root: (profile, "/storage/manifest.json", True, True),
    )
    monkeypatch.setattr(
        llm_staging_module,
        "stage_model_profile",
        lambda *_args, **_kwargs: staged,
    )

    results = llm_staging_module.resolve_and_stage_model_references(
        ["owner/model"],
        tmp_path,
        progress_callback=progress.append,
    )

    assert results[0].requested_reference == "owner/model"
    assert results[0].profile_id == profile.profile_id
    assert results[0].manifest_created is True
    assert results[0].downloaded is True
    assert progress[0].stage == "metadata"
    assert progress[0].message == "Inspecting Hugging Face metadata for owner/model"
    assert progress[0].model_reference == "owner/model"

def test_provider_neutral_stager_uses_planner_resolved_profile(
    llm_staging_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remote staging must not repeat Hugging Face inspection done by the planner."""
    digest = "c" * 64
    raw_profile = {
        "id": f"hf-{digest}",
        "display_name": "Owner Model",
        "repository": "owner/model",
        "revision": "9" * 40,
        "dtype": "bfloat16",
        "modalities": ["text"],
        "estimated_vram_gb": 12,
        "max_context_tokens": 4096,
        "max_images": 1,
        "max_video_frames": 1,
        "max_file_bytes": 1024,
        "max_file_characters": 1024,
        "allow_mixed_image_video": False,
        "trust_remote_code": False,
        "schema_version": 2,
        "source": "generated",
        "profile_digest": digest,
        "backend": "transformers",
        "artifact_bytes": 1024,
    }
    staged = llm_staging_module.StagedModelSnapshot(
        profile_id=raw_profile["id"],
        repository="owner/model",
        revision="9" * 40,
        path=str(tmp_path / "snapshot"),
        downloaded=False,
        elapsed_seconds=0.0,
    )
    monkeypatch.setattr(
        llm_staging_module,
        "stage_model_profile",
        lambda *_args, **_kwargs: staged,
    )
    progress: list[Any] = []

    results = llm_staging_module.resolve_and_stage_model_references(
        ["owner/model"],
        tmp_path,
        resolved_profiles={
            "owner/model": {
                "profile": raw_profile,
                "security_scan_complete": False,
            }
        },
        progress_callback=progress.append,
        owner_id="vast:invocation:owner",
    )

    assert results[0].profile_id == raw_profile["id"]
    assert results[0].security_scan_complete is False
    assert results[0].manifest_created is True
    assert progress[0].stage == "resolved_metadata"
    assert "planner-resolved" in progress[0].message

def test_snapshot_lease_reclaims_dead_local_owner(
    llm_staging_module: Any,
    snapshot_lease_module: Any,
    tmp_path: Path,
) -> None:
    """A dead process record should never force a two-hour stale-lock wait."""
    snapshot_path = tmp_path / "llm_models" / "profile" / ("1" * 40)
    snapshot_path.parent.mkdir(parents=True)
    lease_path = snapshot_path.parent / f".{snapshot_path.name}.download.lock"
    lease_path.write_text(
        json.dumps(
            {
                "owner_id": "orphan",
                "host_id": snapshot_lease_module.socket.gethostname(),
                "pid": 2_000_000_000,
                "process_start": "missing",
                "token": "old-token",
            }
        ),
        encoding="utf-8",
    )

    with llm_staging_module._snapshot_lease(
        snapshot_path,
        model_label="test model",
        owner_id="replacement",
    ):
        owner = json.loads(lease_path.read_text(encoding="utf-8"))
        assert owner["owner_id"] == "replacement"

    assert not lease_path.exists()

def test_snapshot_lease_recognizes_live_owner_without_procfs(
    snapshot_lease_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portable PID probing should keep a live macOS staging owner valid."""
    monkeypatch.setattr(
        snapshot_lease_module,
        "_process_start_identity",
        lambda _pid: None,
    )

    alive = snapshot_lease_module._local_lease_owner_is_alive(
        {
            "host_id": snapshot_lease_module.socket.gethostname(),
            "pid": snapshot_lease_module.os.getpid(),
            "process_start": None,
        }
    )

    assert alive is True

def test_snapshot_lease_reclaims_missing_foreign_heartbeat(
    llm_staging_module: Any,
    snapshot_lease_module: Any,
    tmp_path: Path,
) -> None:
    """A vanished prior container should not leave a structured lease for hours."""
    snapshot_path = tmp_path / "llm_models" / "profile" / ("2" * 40)
    snapshot_path.parent.mkdir(parents=True)
    lease_path = snapshot_path.parent / f".{snapshot_path.name}.download.lock"
    lease_path.write_text(
        json.dumps(
            {
                "owner_id": "old-container",
                "host_id": "different-container",
                "pid": 123,
                "process_start": "456",
                "token": "old-token",
            }
        ),
        encoding="utf-8",
    )
    stale_time = (
        snapshot_lease_module.time.time()
        - snapshot_lease_module._DEFAULT_LEASE_HEARTBEAT_STALE_SECONDS
        - 1
    )
    snapshot_lease_module.os.utime(lease_path, (stale_time, stale_time))

    with llm_staging_module._snapshot_lease(
        snapshot_path,
        model_label="test model",
        owner_id="replacement",
    ):
        owner = json.loads(lease_path.read_text(encoding="utf-8"))
        assert owner["owner_id"] == "replacement"

    assert not lease_path.exists()

def test_snapshot_disk_preflight_rejects_insufficient_capacity(
    llm_staging_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Staging should fail before download when artifacts plus reserve do not fit."""
    profile = llm_staging_module.get_llm_profile("smolvlm2-2.2b-instruct")
    monkeypatch.setattr(
        llm_staging_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=profile.artifact_bytes),
    )

    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        llm_staging_module._preflight_snapshot_capacity(
            tmp_path / "snapshot",
            profile,
        )

def test_direct_stager_payload_encodes_planner_profiles(
    llm_profiles_module: Any,
) -> None:
    """SSH and Vast CLI staging should receive the validated planner envelope."""
    profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")
    entry = {
        "profile": profile.to_mapping(),
        "security_scan_complete": True,
    }

    encoded = llm_profiles_module.encoded_resolved_llm_profile_payloads(
        {"resolved_llm_profiles": {profile.profile_id: entry}},
        [profile.profile_id],
    )

    assert encoded is not None
    decoded = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    assert decoded == {profile.profile_id: entry}

def test_weight_snapshot_is_shared_across_runtime_profiles(
    llm_staging_module: Any,
    tmp_path: Path,
) -> None:
    """Runtime tuning changes must not duplicate one repository revision's weights."""
    calls: list[dict[str, Any]] = []
    base_profile = llm_staging_module.get_llm_profile("smolvlm2-2.2b-instruct")
    throughput_profile = replace(
        base_profile,
        profile_id="smolvlm-throughput-test",
        backend_options=(("enforce_eager", False),),
    )

    def fake_snapshot_download(**kwargs: Any) -> str:
        """Materialize one minimal canonical snapshot."""
        calls.append(kwargs)
        snapshot_path = Path(kwargs["local_dir"])
        snapshot_path.mkdir(parents=True, exist_ok=True)
        (snapshot_path / "config.json").write_text("{}", encoding="utf-8")
        return str(snapshot_path)

    eager = llm_staging_module.stage_model_profile(
        base_profile.profile_id,
        tmp_path,
        profile=base_profile,
        snapshot_download=fake_snapshot_download,
    )
    throughput = llm_staging_module.stage_model_profile(
        throughput_profile.profile_id,
        tmp_path,
        profile=throughput_profile,
        snapshot_download=fake_snapshot_download,
    )

    assert eager.path == throughput.path
    assert eager.downloaded is True
    assert throughput.downloaded is False
    assert len(calls) == 1

def test_stager_reuses_legacy_profile_keyed_weight_snapshot_in_place(
    llm_staging_module: Any,
    tmp_path: Path,
) -> None:
    """Existing profile-keyed weights should remain usable by older deployments."""
    profile = llm_staging_module.get_llm_profile("smolvlm2-2.2b-instruct")
    legacy_path = tmp_path / "llm_models" / profile.profile_id / profile.revision
    legacy_path.mkdir(parents=True)
    (legacy_path / "config.json").write_text("{}", encoding="utf-8")
    (legacy_path / ".comfy-modal-llm-complete.json").write_text(
        json.dumps(
            {
                "profile_id": profile.profile_id,
                "repository": profile.repository,
                "revision": profile.revision,
            }
        ),
        encoding="utf-8",
    )

    result = llm_staging_module.stage_model_profile(
        profile.profile_id,
        tmp_path,
        profile=profile,
        snapshot_download=lambda **kwargs: pytest.fail(
            f"unexpected snapshot download: {kwargs}"
        ),
    )

    assert result.downloaded is False
    assert Path(result.path) == legacy_path
    assert legacy_path.is_dir()

def test_local_curated_profile_is_staged_with_mlx_backend(
    local_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A saved curated id should stage locally without changing the workflow."""
    staged: list[tuple[str, Any, Any]] = []
    monkeypatch.setattr(
        local_llm_runtime_module,
        "stage_model_profile",
        lambda profile_id, storage_root, **kwargs: staged.append(
            (profile_id, Path(storage_root), kwargs["profile"])
        ),
    )

    profile = local_llm_runtime_module.resolve_and_stage_local_profile(
        "smolvlm2-2.2b-instruct",
        tmp_path,
        progress_callback=lambda progress: None,
    )

    assert profile.backend == "mlx_vlm"
    assert profile.execution_target == "local_apple"
    assert staged == [(profile.profile_id, tmp_path, profile)]

def test_remote_dispatch_rewrites_hugging_face_id_to_generated_profile(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU payloads should contain only immutable generated profile IDs."""
    requested_model = "owner/model"
    generated_profile_id = "hf-" + "a" * 64

    class FakeStageMethod:
        """Resolve one user model ID into an immutable profile."""

        def remote(self, model_references: list[str]) -> list[dict[str, Any]]:
            """Return one generated profile result."""
            assert model_references == [requested_model]
            return [
                {
                    "requested_reference": requested_model,
                    "profile_id": generated_profile_id,
                    "revision": "7" * 40,
                    "downloaded": True,
                }
            ]

    class FakeCls:
        """Resolve the CPU ModelStager class."""

        @staticmethod
        def from_name(app_name: str, class_name: str) -> Callable[[], Any]:
            """Return a staging class constructor."""
            assert app_name == "test-b300-app"
            assert class_name == "ModelStager"
            return lambda: SimpleNamespace(stage_profiles=FakeStageMethod())

    monkeypatch.setattr(
        modal_llm_profile_staging_module,
        "modal",
        SimpleNamespace(Cls=FakeCls),
    )
    with modal_llm_profile_staging_module._STAGED_LLM_PROFILES_LOCK:
        modal_llm_profile_staging_module._STAGED_LLM_PROFILES.clear()
        modal_llm_profile_staging_module._STAGED_LLM_PROFILE_RESULTS.clear()
    payload = {
        "component_id": "dynamic-llm",
        "subgraph_prompt": {
            "1": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": requested_model},
            }
        },
    }

    modal_llm_profile_staging_module._ensure_llm_profiles_staged(payload, "test-b300-app")
    direct_inputs = remote_modal_app_module.serialize_node_inputs(
        {"model_profile": requested_model, "prompt": "hello"}
    )
    rewritten_inputs = remote_modal_app_module.deserialize_node_inputs(
        modal_llm_profile_staging_module._rewrite_staged_llm_kwargs_payload(
            direct_inputs,
            "test-b300-app",
        )
    )

    assert (
        payload["subgraph_prompt"]["1"]["inputs"]["model_profile"]
        == generated_profile_id
    )
    assert rewritten_inputs == {
        "model_profile": generated_profile_id,
        "prompt": "hello",
    }
    assert payload["requires_volume_reload"] is True

