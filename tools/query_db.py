#!/usr/bin/env python3
"""
Ascend C 指南数据库查询工具

用法:
    python query_db.py <关键词>
    python query_db.py "ReduceMax"
    python query_db.py "DataCopy"
"""

import sqlite3
import sys
import os

# 查找数据库文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 尝试多个可能的位置
DB_PATHS = [
    os.path.join(PROJECT_ROOT, 'data', 'ascend_c_guide.db'),
    os.path.join(PROJECT_ROOT, 'ascend_c_guide.db'),
    'ascend_c_guide.db',
]

DB_NAME = None
for path in DB_PATHS:
    if os.path.exists(path):
        DB_NAME = path
        break

if DB_NAME is None:
    print('❌ 数据库文件未找到')
    print('   尝试的路径:', DB_PATHS)
    sys.exit(1)

def search(keyword):
    """搜索关键词（统一格式版）"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    print(f'\n🔍 搜索: "{keyword}"')
    print('=' * 50)

    # 优先使用FTS5
    try:
        cursor.execute("""
            SELECT chapter_number, title,
                   snippet(content_fts, 2, '>>>', '<<<', '...', 50) as snippet
            FROM content_fts
            WHERE content_fts MATCH ?
            ORDER BY rank
            LIMIT 10
        """, (keyword,))
        results = cursor.fetchall()
    except:
        results = []

    if not results:
        # 回退LIKE
        print('使用模糊匹配...\n')
        cursor.execute("""
            SELECT chapter_number, title,
                   CASE
                       WHEN instr(lower(content), lower(?)) > 0 THEN
                           '...' || substr(content,
                               max(1, instr(lower(content), lower(?)) - 25),
                               100) || '...'
                       ELSE substr(content, 1, 100) || '...'
                   END as snippet
            FROM chapters
            WHERE title LIKE ? OR content LIKE ?
            LIMIT 10
        """, (keyword, keyword, f'%{keyword}%', f'%{keyword}%'))
        results = cursor.fetchall()

    # 统一输出格式
    if results:
        for idx, (chapter, title, snippet) in enumerate(results, 1):
            print(f'\n【{idx}】 {chapter} - {title}')
            print(f'    {snippet}')
    else:
        print('\n未找到相关内容')

    conn.close()

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    keyword = ' '.join(sys.argv[1:])
    search(keyword)

if __name__ == '__main__':
    main()
