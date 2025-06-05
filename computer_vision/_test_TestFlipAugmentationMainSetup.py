import pytest
import os
import shutil
import glob
import random
from string import ascii_lowercase, digits
import cv2
import numpy
from _test_FlipAugmentationMain import Test_FlipAugmentationMain

class Test_TestFlipAugmentationMainSetup:
    @pytest.fixture(scope="function", autouse=True)
    def setup_teardown(self):
        self.test_instance = Test_FlipAugmentationMain()
        self.test_instance.setup()

    @pytest.mark.smoke
    def test_directories_creation_and_cleanup(self):
        assert os.path.exists(self.test_instance.label_dir)
        assert os.path.exists(self.test_instance.img_dir)
        assert os.path.exists(self.test_instance.output_dir)
        shutil.rmtree(self.test_instance.label_dir)
        shutil.rmtree(self.test_instance.img_dir)
        shutil.rmtree(self.test_instance.output_dir)
        assert not os.path.exists(self.test_instance.label_dir)
        assert not os.path.exists(self.test_instance.img_dir)
        assert not os.path.exists(self.test_instance.output_dir)

    @pytest.mark.positive
    def test_directory_existence(self):
        assert os.path.exists(self.test_instance.label_dir)
        assert os.path.exists(self.test_instance.img_dir)
        assert os.path.exists(self.test_instance.output_dir)
        # Cleanup directories to simulate end of test
        shutil.rmtree(self.test_instance.label_dir)
        shutil.rmtree(self.test_instance.img_dir)
        shutil.rmtree(self.test_instance.output_dir)
        assert not os.path.exists(self.test_instance.label_dir)
        assert not os.path.exists(self.test_instance.img_dir)
        assert not os.path.exists(self.test_instance.output_dir)

    @pytest.mark.positive
    def test_directory_permissions(self):
        assert os.access(self.test_instance.label_dir, os.R_OK | os.W_OK | os.X_OK)
        assert os.access(self.test_instance.img_dir, os.R_OK | os.W_OK | os.X_OK)
        assert os.access(self.test_instance.output_dir, os.R_OK | os.W_OK | os.X_OK)

    @pytest.mark.positive
    def test_concurrent_directory_creation(self):
        def create_and_cleanup():
            test_instance = Test_FlipAugmentationMain()
            test_instance.setup()
            assert os.path.exists(test_instance.label_dir)
            assert os.path.exists(test_instance.img_dir)
            assert os.path.exists(test_instance.output_dir)
            shutil.rmtree(test_instance.label_dir)
            shutil.rmtree(test_instance.img_dir)
            shutil.rmtree(test_instance.output_dir)
            assert not os.path.exists(test_instance.label_dir)
            assert not os.path.exists(test_instance.img_dir)
            assert not os.path.exists(test_instance.output_dir)

        # Run multiple times concurrently to simulate concurrent tests
        pytest.helpers.run_concurrent(create_and_cleanup, create_and_cleanup, create_and_cleanup)

    @pytest.mark.negative
    def test_directory_cleanup_on_exception(self):
        try:
            raise Exception("Intentional Exception")
        except Exception:
            pass
        assert not os.path.exists(self.test_instance.label_dir)
        assert not os.path.exists(self.test_instance.img_dir)
        assert not os.path.exists(self.test_instance.output_dir)

    @pytest.mark.positive
    def test_directory_naming_conventions(self):
        assert self.test_instance.label_dir == 'test_labels'
        assert self.test_instance.img_dir == 'test_images'
        assert self.test_instance.output_dir == 'test_output'
