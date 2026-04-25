export function encodePickIndexToRgb(pickIndex) {
  const resolvedPickIndex = Number(pickIndex);
  if (!Number.isInteger(resolvedPickIndex) || resolvedPickIndex < 0 || resolvedPickIndex > 0xffffff) {
    throw new RangeError("pickIndex must be an integer between 0 and 16777215");
  }
  return [
    resolvedPickIndex & 255,
    (resolvedPickIndex >> 8) & 255,
    (resolvedPickIndex >> 16) & 255,
  ];
}

export function decodePickIndexFromRgb(red, green, blue) {
  return (red & 255) + ((green & 255) << 8) + ((blue & 255) << 16);
}

export function findNearestPickHit({
  pixels,
  width,
  height,
  originX,
  originY,
  targetX,
  targetY,
}) {
  let bestHit = null;
  let bestDistanceSq = Number.POSITIVE_INFINITY;

  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const offset = (row * width + column) * 4;
      const pickIndex = decodePickIndexFromRgb(
        pixels[offset + 0],
        pixels[offset + 1],
        pixels[offset + 2],
      );
      if (pickIndex === 0) {
        continue;
      }

      const pixelX = originX + column;
      const pixelY = originY + row;
      const distanceSq = (pixelX - targetX) ** 2 + (pixelY - targetY) ** 2;
      if (distanceSq >= bestDistanceSq) {
        continue;
      }

      bestDistanceSq = distanceSq;
      bestHit = {
        pickIndex,
        screenX: pixelX,
        screenY: pixelY,
        pixelDistance: Math.sqrt(distanceSq),
      };
    }
  }

  return bestHit;
}
