from huggingface_hub import get_collection, list_repo_files, hf_hub_download

import pytest
import json
import numpy as np

from .versioned_fixtures import versioned_unhashable_object_fixture, versioned_hashable_object_fixture


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


def get_hf_model_file(cache, repo_id, filename):
    
    cache_dir = cache.mkdir('hf_cache')

    file_name = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir)
    )

    return str(file_name)


def get_hf_dataset_file(cache, repo_id, filename):
    
    cache_dir = cache.mkdir('hf_cache')

    file_name = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(cache_dir)
    )

    return str(file_name)


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
