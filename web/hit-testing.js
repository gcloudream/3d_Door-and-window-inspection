export function computeSquareHitMetrics({
  screenX,
  screenY,
  pointScreenX,
  pointScreenY,
  halfSizePx,
}) {
  const dx = pointScreenX - screenX;
  const dy = pointScreenY - screenY;
  const absDx = Math.abs(dx);
  const absDy = Math.abs(dy);
  const resolvedHalfSizePx = Math.max(0, halfSizePx);
  const squareDistance = Math.max(absDx, absDy);
  const pixelDistance = Math.hypot(dx, dy);
  const insideSquare = absDx <= resolvedHalfSizePx && absDy <= resolvedHalfSizePx;

  return {
    dx,
    dy,
    pixelDistance,
    squareDistance,
    insideSquare,
    effectiveSquareDistance: Math.max(0, squareDistance - resolvedHalfSizePx),
  };
}

export function buildHintCandidateKey({
  effectiveSquareDistance,
  pixelDistance,
  depth,
  projectionLength = Number.POSITIVE_INFINITY,
  distanceToRay = Number.POSITIVE_INFINITY,
  pointId,
  insideSquare = false,
}) {
  if (insideSquare) {
    return [
      effectiveSquareDistance,
      0,
      projectionLength,
      depth,
      distanceToRay,
      pixelDistance,
      pointId,
    ];
  }

  return [
    effectiveSquareDistance,
    1,
    pixelDistance,
    depth,
    projectionLength,
    distanceToRay,
    pointId,
  ];
}
