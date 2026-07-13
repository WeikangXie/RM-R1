#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据合并脚本
功能：将comment数据和post数据合并，生成完整的训练数据
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# comment_data 中保留的评论字段
COMMENT_FIELDS_TO_KEEP = (
    'commentId',
    'commentContent',
    'commentState',
    'extendType',
    'commentType',
    'parentType',
)

# 需要从post中合并的字段
FIELDS_TO_MERGE = (
    'text',
    'productName',
    'topicList',
    'relatedCoterieList',
)

LABEL_BY_COMMENT_STATE = {
    'PUBLISHED': 'pass',
    'HIDE': 'reject',
}

class DataCombiner:
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.raw_comment_file = self.base_dir / 'oa_export_test_comment.jsonl'
        self.post_file = self.base_dir / 'oa_export_test_post.jsonl'
        self.combine_file = self.base_dir / 'combine_data.jsonl'
        self.filtered_comment_file = self.base_dir / 'comment_data.jsonl'
        self.dirty_file = self.base_dir / 'dirty_data.jsonl'
        self.post_map = {}

    def load_post_data(self):
        """加载post数据到内存字典中，以postId为键"""
        logger.info(f"开始加载post数据: {self.post_file}")

        if not self.post_file.exists():
            logger.error(f"post文件不存在: {self.post_file}")
            return

        with open(self.post_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    post_data = json.loads(line)
                    post_id = post_data.get('postId')
                    if post_id:
                        self.post_map[post_id] = post_data
                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_num}行JSON解析失败: {e}")

        logger.info(f"post数据加载完成，共 {len(self.post_map)} 条记录")

    def filter_comment_data(self, comment_data):
        """只保留项目消费的 comment 字段。"""
        if comment_data.get('auditState') != "OPERATOR_AUDITED":
            return None

        return {
            field: comment_data[field]
            for field in COMMENT_FIELDS_TO_KEEP
            if field in comment_data
        }

    def merge_data(self):
        """合并comment和post数据"""
        logger.info(f"开始合并数据: {self.raw_comment_file}")

        if not self.raw_comment_file.exists():
            logger.error(f"comment文件不存在: {self.raw_comment_file}")
            return

        combined_count = 0
        dirty_count = 0

        with open(self.combine_file, 'w', encoding='utf-8') as combined_f, \
             open(self.dirty_file, 'w', encoding='utf-8') as dirty_f:

            for line_num, line in enumerate(self.raw_comment_file.open('r', encoding='utf-8'), 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    comment_data = json.loads(line)

                    # 过滤数据和指定字段
                    filtered_data = self.filter_comment_data(comment_data)
                    if not filtered_data:
                        continue

                    # 获取parentId
                    parent_id = comment_data.get('parentId')

                    if not parent_id:
                        # parentId为空，写入dirty数据
                        dirty_f.write(json.dumps(filtered_data, ensure_ascii=False) + '\n')
                        dirty_count += 1
                        continue

                    # 查找对应的post数据
                    post_data = self.post_map.get(parent_id)

                    if not post_data:
                        # 找不到对应的post，写入dirty数据
                        dirty_f.write(json.dumps(filtered_data, ensure_ascii=False) + '\n')
                        dirty_count += 1
                        continue

                    # 合并post数据中需要的字段
                    for field in FIELDS_TO_MERGE:
                        if field in post_data:
                            filtered_data[field] = post_data[field]

                    # 写入合并后的数据
                    combined_f.write(json.dumps(filtered_data, ensure_ascii=False) + '\n')
                    combined_count += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_num}行JSON解析失败: {e}")

        logger.info(f"数据合并完成: 成功合并 {combined_count} 条, 脏数据 {dirty_count} 条")

    def filter_merge_data(self, output_file=None):
        """统计合并数据分布，并将全部数据写入 comment_data。

        Args:
            output_file: 输出文件路径，默认在base_dir下创建comment_data.jsonl
        """
        if output_file is None:
            output_file = self.filtered_comment_file
        else:
            output_file = Path(output_file)

        if not self.combine_file.exists():
            logger.error(f"合并后的数据文件不存在: {self.combine_file}")
            return

        logger.info("开始统计合并数据")

        # 读取所有数据
        all_data = []
        with open(self.combine_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析失败: {e}")

        logger.info(f"总共读取到 {len(all_data)} 条数据")

        comment_state_counts = Counter()
        audit_label_counts = Counter()
        extend_type_counts = Counter()
        extend_type_audit_label_counts = defaultdict(Counter)
        product_names = set()

        for data in all_data:
            comment_state = data.get('commentState') or 'UNKNOWN'
            audit_label = LABEL_BY_COMMENT_STATE.get(comment_state, 'unknown')
            extend_type = data.get('extendType') or 'UNKNOWN'
            product_name = data.get('productName')

            comment_state_counts[comment_state] += 1
            audit_label_counts[audit_label] += 1
            extend_type_counts[extend_type] += 1
            extend_type_audit_label_counts[extend_type][audit_label] += 1
            if product_name:
                product_names.add(product_name)

        summary = {
            'total': len(all_data),
            'comment_state_counts': dict(comment_state_counts),
            'audit_label_counts': dict(audit_label_counts),
            'extend_type_counts': dict(extend_type_counts),
            'extend_type_audit_label_counts': {
                extend_type: dict(label_counts)
                for extend_type, label_counts in extend_type_audit_label_counts.items()
            },
            'distinct_product_count': len(product_names),
        }
        logger.info(
            "数据分布统计: %s",
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
        )

        with open(output_file, 'w', encoding='utf-8') as combined_f:
            for data in all_data:
                combined_f.write(json.dumps(data, ensure_ascii=False) + '\n')

        return summary


    def main(self):
        """主函数"""
        try:
            self.load_post_data()
            self.merge_data()
            self.filter_merge_data()
        except Exception as e:
            logger.error(f"程序执行失败: {e}")

if __name__ == "__main__":
    combiner = DataCombiner(base_dir='./data/local/raw')
    combiner.main()
