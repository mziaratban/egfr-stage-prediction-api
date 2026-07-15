import time

from starlette.middleware.base import BaseHTTPMiddleware

from .metrics import (
    active_predictions,
    prediction_duration,
)

class PredictionMetricsMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):

        if request.url.path != "/predict":
            return await call_next(request)

        active_predictions.inc()

        start = time.perf_counter()

        try:

            response = await call_next(request)

            return response

        finally:

            elapsed = time.perf_counter() - start

            prediction_duration.observe(elapsed)

            active_predictions.dec()