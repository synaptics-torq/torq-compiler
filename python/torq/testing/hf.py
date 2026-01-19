from huggingface_hub import get_collection, list_repo_files, hf_hub_download, try_to_load_from_cache, get_hf_file_metadata, hf_hub_url

import pytest
import json
import numpy as np

from .versioned_fixtures import versioned_unhashable_object_fixture, versioned_hashable_object_fixture
from . import s3

def find_mlir_files_from_repo(cache, repo_id):

    cache_key = "hf_list_repo" + repo_id

    cached_data = cache.get(cache_key, None)

    # Return cached data if available
    if cached_data is not None:
        return json.loads(cached_data)
    
    # Fetch data from the Hugging Face Hub
    model_files = list_repo_files(repo_id=repo_id)

    result = [(repo_id, file) for file in model_files if file.endswith(".mlir")]

    # Cache the result for future test runs
    cache.set(cache_key, json.dumps(result))

    return result

    

def find_mlir_files_from_collection(cache, collection_id):
    """
    Find all the test models in the Hugging Face Hub in a given collection
    """
    
    cache_key = "hf_list_collection" + collection_id

    cached_data = cache.get(cache_key, None)

    # Return cached data if available
    if cached_data is not None:
        return json.loads(cached_data)

    # Fetch data from the Hugging Face Hub
    collection = get_collection(collection_id)

    models = [ x for x in collection.items if x.item_type == "model"]

    result = []

    for model in models:
        result.extend(find_mlir_files_from_repo(cache, model.item_id))

    # Cache the result for future test runs
    cache.set(cache_key, json.dumps(result))

    return result


def generate_test_for_collection(metafunc, fixture_name, collection_id):

    if fixture_name in metafunc.fixturenames:
        cache = metafunc.config.cache
        params = find_mlir_files_from_collection(cache, fixture_name, collection_id)
        metafunc.parametrize(
            fixture_name,
            params,
            ids=[f"{param[0]}::{param[1]}" for param in params],
            indirect=True
        )


def get_hf_file_with_s3_cache(cache, repo_id, filename, repo_type="model"):
    """
    Internal helper to download files from HuggingFace Hub with three-tier caching:
    1. Check local cache
    2. Try S3 cache (using etag)
    3. Fall back to HuggingFace Hub download (and populate S3 cache)
    """

    category = repo_type

    # fetch the etag and revision of the latest version
    try:
        url = hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type
        )

        metadata = get_hf_file_metadata(url=url)
        etag = metadata.etag
        revision = metadata.commit_hash
    except Exception:
        raise RuntimeError(f"Failed to get metadata for {repo_id}/{filename}")

    # try to get the file from the S3 cache (either the local dir or the S3 bucket)
    print("Trying to get file from S3 cache:", repo_id, filename, etag)

    file_name = s3.get_file(
        cache=cache,
        category=category,
        etag=etag
    )

    if file_name is not None:
        print("Found file in S3 cache:", file_name)
        return str(file_name)
    
    # file not found in the s3 cache, download from HuggingFace Hub 
    cache_dir = cache.mkdir('hf_cache')
    file_name = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
        repo_type=repo_type,
        revision=revision
    )
    
    # try to update S3 cache with the downloaded file
    file_name = s3.put_file(
        cache=cache,
        category=category,
        etag=etag,
        local_file_path=file_name
    )

    return str(file_name)


def get_hf_model_file(cache, repo_id, filename):
    return get_hf_file_with_s3_cache(cache, repo_id, filename, repo_type="model")


def get_hf_dataset_file(cache, repo_id, filename):
    return get_hf_file_with_s3_cache(cache, repo_id, filename, repo_type="dataset")


@versioned_hashable_object_fixture
def hf_model_name():
    return {"repo_id": "Synaptics/TorqCompilerTestModels", "filename": "mobilenet_v2_1.0_224.mlir"}


@versioned_unhashable_object_fixture
def hf_model(request, hf_model_name):
    """
    Fixture to return the Hugging Face model file
    """
    
    cache = request.getfixturevalue("cache")

    return get_hf_model_file(cache, hf_model_name["repo_id"], hf_model_name["filename"])


@pytest.fixture
def hf_image_name():
    return {"repo_id": "Synaptics/TorqCompilerTestImages", "filename": "space_shuttle_224x224.jpg"}


@versioned_unhashable_object_fixture
def hf_image(request, hf_image_name):
    """
    Fixture to return a sample imagenet image from ImageNet
    """    

    image_path = get_hf_dataset_file(request.getfixturevalue("cache"), hf_image_name["repo_id"], hf_image_name["filename"])

    from PIL import Image

    image = Image.open(image_path)

    class FakeImage:
        def numpy(self):
            return np.array(image, dtype=np.uint8)

    return FakeImage()


@versioned_hashable_object_fixture
def hf_audio_name():
    return {"repo_id": "Synaptics/TorqCompilerTestAudio", "filename": "apostle.wav", "transcription_filename": "apostle.wav.txt"}


@versioned_unhashable_object_fixture
def hf_audio_with_transcription(request, hf_audio_name):
    """
    Fixture to return a sample audio snipped with its transcription
    """    

    cache = request.getfixturevalue("cache")

    audio_file = get_hf_dataset_file(cache, hf_audio_name["repo_id"], hf_audio_name["filename"])
    transcription_file = get_hf_dataset_file(cache, hf_audio_name["repo_id"], hf_audio_name["transcription_filename"])

    import wave

    with wave.open(audio_file, 'rb') as wf:
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

    with open(transcription_file, 'r') as fp:
        transcription = fp.read().strip()

    return audio_np, transcription


@versioned_hashable_object_fixture
def model_large_512_stream_bf16_onnx_name():
    return {"repo_id": "Synaptics/Customer_A", "filename": "model_large_512_stream_bf16.onnx"}


@versioned_unhashable_object_fixture
def model_large_512_stream_bf16_onnx(request, model_large_512_stream_bf16_onnx_name):

    cache = request.getfixturevalue("cache")

    return get_hf_model_file(cache,
        model_large_512_stream_bf16_onnx_name["repo_id"], model_large_512_stream_bf16_onnx_name["filename"])
