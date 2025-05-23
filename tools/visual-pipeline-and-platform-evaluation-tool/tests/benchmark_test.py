import unittest
from unittest.mock import patch

from benchmark import (
    Benchmark,
)

from pipeline import SmartNVRPipeline


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.video_path = "test_video.mp4"
        self.pipeline_cls = SmartNVRPipeline
        self.fps_floor = 30.0
        self.rate = 50
        self.parameters = {"object_detection_device": "cpu"}
        self.constants = {"const1": "value1"}
        self.elements = [("element1", "type1", "name1")]
        self.benchmark = Benchmark(
            video_path=self.video_path,
            pipeline_cls=self.pipeline_cls,
            fps_floor=self.fps_floor,
            rate=self.rate,
            parameters=self.parameters,
            constants=self.constants,
            elements=self.elements,
        )

    def test_run(self):
        with patch.object(Benchmark, "_run_pipeline_and_extract_metrics") as mock_run:
            mock_run.side_effect = [
                # First call with 1 stream
                [
                    {
                        "params": {},
                        "exit_code": 0,
                        "total_fps": 30,
                        "per_stream_fps": 30,
                        "num_streams": 1,
                    }
                ],
                # Second call with 2 streams
                [
                    {
                        "params": {},
                        "exit_code": 0,
                        "total_fps": 168,
                        "per_stream_fps": 28,
                        "num_streams": 6,
                    }
                ],
                # Third call with 3 streams
                [
                    {
                        "params": {},
                        "exit_code": 0,
                        "total_fps": 155,
                        "per_stream_fps": 31,
                        "num_streams": 5,
                    }
                ],
            ]
            result = self.benchmark.run()
            self.assertEqual(result, (5, 3, 2, 31))


if __name__ == "__main__":
    unittest.main()
