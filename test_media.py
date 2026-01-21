#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试媒体文件是否可以正常访问
"""

import os
from PIL import Image

def test_media_files():
    """测试媒体文件"""
    print("🎨 媒体文件测试")
    print("=" * 40)

    # 测试图片文件
    image_path = "asset/game_interface.jpg"
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            width, height = img.size
            size_kb = os.path.getsize(image_path) / 1024
            print(f"✅ 图片文件: {image_path}")
            print(f"   尺寸: {width} x {height} 像素")
            print(f"   大小: {size_kb:.1f} KB")
            print(f"   格式: {img.format}")
        except Exception as e:
            print(f"❌ 图片文件无法打开: {e}")
    else:
        print(f"❌ 图片文件不存在: {image_path}")

    # 测试视频文件
    video_path = "asset/game_demo.mp4"
    if os.path.exists(video_path):
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"✅ 视频文件: {video_path}")
        print(f"   大小: {size_mb:.1f} MB")
        print(f"   格式: MP4")
    else:
        print(f"❌ 视频文件不存在: {video_path}")

    # 检查 README 中的引用
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "game_interface.jpg" in content:
                print("✅ README 中正确引用了图片文件")
            else:
                print("❌ README 中未找到图片引用")

            if "game_demo.mp4" in content:
                print("✅ README 中正确引用了视频文件")
            else:
                print("❌ README 中未找到视频引用")

    print("\n🎉 媒体文件测试完成！")
    print("README.md 中的图片和视频现在应该可以正常显示了！")

if __name__ == "__main__":
    test_media_files()

