import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:camera_platform_interface/camera_platform_interface.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:fruit_recognition_app/preprocess.dart';

/// 8x4 YUV420 frame whose Y plane value is (row * 8 + column).
CameraImage _frame() {
  final y = Uint8List(8 * 4);
  for (var i = 0; i < y.length; i++) {
    y[i] = i;
  }
  final uv = Uint8List(4 * 2)..fillRange(0, 8, 128);

  return CameraImage.fromPlatformInterface(
    CameraImageData(
      format: const CameraImageFormat(
        ImageFormatGroup.yuv420,
        raw: 35,
      ),
      width: 8,
      height: 4,
      planes: [
        CameraImagePlane(bytes: y, bytesPerRow: 8, bytesPerPixel: 1),
        CameraImagePlane(bytes: uv, bytesPerRow: 4, bytesPerPixel: 1),
        CameraImagePlane(bytes: Uint8List.fromList(uv), bytesPerRow: 4, bytesPerPixel: 1),
      ],
    ),
  );
}

void main() {
  test('keeps raw 0-255 values (no mean/std normalization)', () {
    final y = Uint8List(8 * 4)..fillRange(0, 32, 200);
    final uv = Uint8List(4 * 2)..fillRange(0, 8, 128);
    final image = CameraImage.fromPlatformInterface(
      CameraImageData(
        format: const CameraImageFormat(ImageFormatGroup.yuv420, raw: 35),
        width: 8,
        height: 4,
        planes: [
          CameraImagePlane(bytes: y, bytesPerRow: 8, bytesPerPixel: 1),
          CameraImagePlane(bytes: uv, bytesPerRow: 4, bytesPerPixel: 1),
          CameraImagePlane(bytes: Uint8List.fromList(uv), bytesPerRow: 4, bytesPerPixel: 1),
        ],
      ),
    );

    final out = Float32List(kImageSize * kImageSize * 3);
    yuvToModelInput(out, image, 0);

    expect(out.every((v) => v == 200), isTrue);
  });

  test('rotation 90 maps the upright frame back to the sensor frame', () {
    final out = Float32List(kImageSize * kImageSize * 3);
    yuvToModelInput(out, _frame(), 90);

    // Top-left of the upright frame is the sensor's bottom-left pixel.
    expect(out.sublist(0, 3), [3 * 8 + 0, 3 * 8 + 0, 3 * 8 + 0]);
    // Bottom-right is the sensor's top-right pixel.
    expect(out.sublist(out.length - 3), [0 * 8 + 7, 0 * 8 + 7, 0 * 8 + 7]);
  });
}
