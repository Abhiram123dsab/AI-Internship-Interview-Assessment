import sys
import os
import unittest

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the test module
from test_comprehensive import TestMultiLanguageCommunicationSystem

if __name__ == '__main__':
    # Create a test suite with all tests from TestMultiLanguageCommunicationSystem
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiLanguageCommunicationSystem)
    
    # Run the tests
    unittest.TextTestRunner(verbosity=2).run(suite)