import unittest
import subprocess
import sys
from constants import DEFAULT_LOCAL_BASE_MODEL_PATH, DEFAULT_ADAPTER_DIR


class TestInfer(unittest.TestCase):
    def test_infer_list_files_command(self):
        """Test that infer.py returns a valid command for listing files."""
        # Run infer.py as a subprocess
        result = subprocess.run(
            [
                sys.executable,
                "infer.py",
                "--base_model", DEFAULT_LOCAL_BASE_MODEL_PATH,
                "--adapter", DEFAULT_ADAPTER_DIR,
                "--prompt", "list all files in current directory"
            ],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout for model loading
        )
        
        # Check that the command ran successfully
        self.assertEqual(result.returncode, 0, f"infer.py failed with error: {result.stderr}")
        
        # Get the output command
        output = result.stdout.strip()
        
        # Assert that we got a non-empty response
        self.assertTrue(len(output) > 0, "Expected a command but got empty output")
        
        # Assert that the output looks like a command (not a blocked message)
        self.assertNotIn("Blocked potentially destructive", output)
        
        # Assert that it's likely an ls command
        self.assertIn("ls", output.lower(), f"Expected ls command but got: {output}")
        
        # Assert it's a single line (no newlines)
        self.assertNotIn("\n", output, f"Expected single line command but got multiple lines: {output}")


if __name__ == "__main__":
    unittest.main()