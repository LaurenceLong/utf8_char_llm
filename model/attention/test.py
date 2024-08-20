import unittest
import torch

from model.attention.dcba_attention import DynamicConvolutionalBlockedAttention


class TestPBDCA(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.nhead = 8
        self.block_size = 64
        self.seq_len = 2048
        self.batch_size = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DynamicConvolutionalBlockedAttention(self.d_model, self.nhead, self.block_size).to(self.device)

    def test_dcba_output_shape(self):
        # 准备输入数据
        x = torch.randn(self.batch_size, self.seq_len, self.d_model).to(self.device)

        # 运行模型
        output = self.model(x)

        # 检查输出形状
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape,
                         f"预期输出形状为 {expected_shape}，但得到 {output.shape}")

        self.assertIsInstance(output, torch.Tensor, "输出应该是 torch.Tensor 类型")
        print(output.shape)


if __name__ == '__main__':
    unittest.main()
