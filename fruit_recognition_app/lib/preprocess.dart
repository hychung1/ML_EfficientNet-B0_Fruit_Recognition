import 'dart:typed_data';

import 'package:camera/camera.dart';

const kImageSize = 224;

/// Samples a YUV420 camera frame into the 224x224 RGB float buffer the model
/// expects. [rotation] is the clockwise rotation (0/90/180/270) that makes the
/// frame upright.
///
/// Values are raw 0-255: Keras 3's EfficientNetB0 contains its own rescaling
/// and normalization layers, so the model input is not pre-normalized.
void yuvToModelInput(Float32List out, CameraImage image, int rotation) {
  final planes = image.planes;
  if (planes.length < 3) {
    throw ArgumentError('Expected a YUV420 frame with 3 planes');
  }

  final w = image.width;
  final h = image.height;
  final yPlane = planes[0];
  final uPlane = planes[1];
  final vPlane = planes[2];
  final yRowStride = yPlane.bytesPerRow;
  final yPixelStride = yPlane.bytesPerPixel ?? 1;
  final uRowStride = uPlane.bytesPerRow;
  final uPixelStride = uPlane.bytesPerPixel ?? 1;
  final vRowStride = vPlane.bytesPerRow;
  final vPixelStride = vPlane.bytesPerPixel ?? 1;

  // Size of the frame after rotating it upright.
  final rotatedWidth = rotation % 180 == 0 ? w : h;
  final rotatedHeight = rotation % 180 == 0 ? h : w;

  var p = 0;
  for (var oy = 0; oy < kImageSize; oy++) {
    final ry = ((oy + 0.5) * rotatedHeight ~/ kImageSize)
        .clamp(0, rotatedHeight - 1);
    for (var ox = 0; ox < kImageSize; ox++) {
      final rx = ((ox + 0.5) * rotatedWidth ~/ kImageSize)
          .clamp(0, rotatedWidth - 1);
      // Map the upright pixel back to the raw sensor pixel.
      final (sx, sy) = switch (rotation) {
        90 => (ry, h - 1 - rx),
        180 => (w - 1 - rx, h - 1 - ry),
        270 => (w - 1 - ry, rx),
        _ => (rx, ry),
      };

      final y = yPlane.bytes[sy * yRowStride + sx * yPixelStride];
      final u =
          uPlane.bytes[(sy >> 1) * uRowStride + (sx >> 1) * uPixelStride] - 128;
      final v =
          vPlane.bytes[(sy >> 1) * vRowStride + (sx >> 1) * vPixelStride] - 128;

      out[p++] = (y + 1.402 * v).clamp(0, 255);
      out[p++] = (y - 0.344136 * u - 0.714136 * v).clamp(0, 255);
      out[p++] = (y + 1.772 * u).clamp(0, 255);
    }
  }
}
