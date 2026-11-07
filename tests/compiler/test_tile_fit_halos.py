"""Position-dependent producer footprints must not escape tile-fit validation."""

import subprocess


def fused_halo_model():
    # Two vertical radius-3 halos and one horizontal radius-3 halo. A 3x5
    # consumer tile at (9,10) needs 14x10 upstream, but (6,5) needs 15x11.
    lines = [
        "module {",
        "  func.func @main(%input: tensor<1x17x17x768xi8>) -> tensor<1x17x17x160xi8> {",
        '    %zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>',
        '    %z32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>',
        '    %m = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>',
        '    %s = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>',
        '    %bias = "tosa.const"() <{values = dense<0> : tensor<160xi32>}> : () -> tensor<160xi32>',
    ]
    value = "%input"
    for i, (channels, kh, kw) in enumerate([(768, 1, 1), (160, 7, 1), (160, 1, 7), (160, 7, 1)]):
        weight = f"tensor<160x{kh}x{kw}x{channels}xi8>"
        src = f"tensor<1x17x17x{channels}xi8>"
        acc = "tensor<1x17x17x160xi32>"
        dst = "tensor<1x17x17x160xi8>"
        lines += [
            f'    %w{i} = "tosa.const"() <{{values = dense<1> : {weight}}}> : () -> {weight}',
            f"    %c{i} = tosa.conv2d {value}, %w{i}, %bias, %zp, %zp "
            f"pad([{kh // 2}, {kh // 2}, {kw // 2}, {kw // 2}]) stride([1, 1]) "
            f"dilation([1, 1]) acc_type(i32) : ({src}, {weight}, tensor<160xi32>, "
            f"tensor<1xi8>, tensor<1xi8>) -> {acc}",
            f"    %r{i} = tosa.rescale %c{i}, %m, %s, %z32, %zp "
            "scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) "
            "input_unsigned(false) output_unsigned(false) : "
            f"({acc}, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> {dst}",
        ]
        value = f"%r{i}"
    lines += [f"    return {value} : tensor<1x17x17x160xi8>", "  }", "}"]
    return "\n".join(lines)


def test_fused_halo_tiles_fit_at_every_position(torq_compiler, tmp_path):
    model = tmp_path / "fused_halos.mlir"
    model.write_text(fused_halo_model())
    proc = subprocess.run(
        [
            str(torq_compiler.file_path), str(model), "-o", str(tmp_path / "model.vmfb"),
            "--torq-hw=1:256:1:coral_v2:", "--torq-target-host-triple=native",
            "--torq-disable-host", "--torq-disable-css", "--mlir-disable-threading",
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "model.vmfb").stat().st_size > 0