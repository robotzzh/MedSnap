# -*- coding: utf-8 -*-
"""
MedSnap 统计分析 — Flask Blueprint 路由
所有 /api/stats/* 路由和 /statistics 页面路由。
"""

import json
import sqlite3
import os
import io
from flask import Blueprint, render_template, request, jsonify, send_file, current_app

from statistics_engine import (
    DatasetManager, Preprocessor, DescriptiveAnalyzer,
    ComparisonAnalyzer, RegressionAnalyzer,
    PrerequisiteTests, ChartGenerator, ExportManager,
    get_dataset, register_dataset, remove_dataset
)

stats_bp = Blueprint('stats', __name__)


# ==============================================================================
#  辅助函数
# ==============================================================================

def _get_db():
    """获取数据库连接（复用 app.py 的 DB_PATH）"""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_medical_data.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ==============================================================================
#  页面路由
# ==============================================================================

@stats_bp.route('/statistics')
def statistics_page():
    return render_template('statistics.html')


# ==============================================================================
#  数据集管理路由
# ==============================================================================

@stats_bp.route('/api/stats/upload', methods=['POST'])
def stats_upload():
    """上传 CSV/Excel 文件创建数据集"""
    try:
        f = request.files.get('file')
        if not f or not f.filename:
            return jsonify({"status": "error", "msg": "请选择文件"})
        fname = f.filename.lower()
        if fname.endswith('.csv') or fname.endswith('.txt'):
            dm = DatasetManager.from_csv(f, filename=f.filename)
        elif fname.endswith(('.xlsx', '.xls')):
            dm = DatasetManager.from_excel(f, filename=f.filename)
        else:
            return jsonify({"status": "error", "msg": "不支持的文件格式，请上传 CSV 或 Excel 文件"})
        dataset_id = register_dataset(dm)
        return jsonify({"status": "success", "dataset_id": dataset_id, "info": dm.get_info()})
    except Exception as e:
        return jsonify({"status": "error", "msg": f"文件解析失败: {str(e)}"})


@stats_bp.route('/api/stats/import_records', methods=['POST'])
def stats_import_records():
    """从已有病历记录导入数据"""
    try:
        data = request.get_json()
        record_ids = data.get('record_ids', [])
        role_id = data.get('role_id', '')

        conn = _get_db()
        c = conn.cursor()
        if record_ids:
            placeholders = ','.join(['?'] * len(record_ids))
            c.execute(f'SELECT extracted_data FROM medical_records WHERE id IN ({placeholders})', record_ids)
        elif role_id:
            c.execute('SELECT extracted_data FROM medical_records WHERE role_id=?', (role_id,))
        else:
            c.execute('SELECT extracted_data FROM medical_records')
        rows = [dict(r) for r in c.fetchall()]
        conn.close()

        if not rows:
            return jsonify({"status": "error", "msg": "未找到符合条件的记录"})

        dm = DatasetManager.from_records(rows)
        dataset_id = register_dataset(dm)
        return jsonify({"status": "success", "dataset_id": dataset_id, "info": dm.get_info()})
    except Exception as e:
        return jsonify({"status": "error", "msg": f"导入失败: {str(e)}"})


@stats_bp.route('/api/stats/dataset/<dataset_id>')
def stats_dataset_info(dataset_id):
    dm = get_dataset(dataset_id)
    if not dm:
        return jsonify({"status": "error", "msg": "数据集不存在或已过期"})
    return jsonify({"status": "success", "info": dm.get_info()})


@stats_bp.route('/api/stats/dataset/<dataset_id>/preview')
def stats_dataset_preview(dataset_id):
    dm = get_dataset(dataset_id)
    if not dm:
        return jsonify({"status": "error", "msg": "数据集不存在"})
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 50, type=int)
    return jsonify({"status": "success", "preview": dm.get_preview(page, page_size)})


@stats_bp.route('/api/stats/columns/<dataset_id>')
def stats_columns(dataset_id):
    dm = get_dataset(dataset_id)
    if not dm:
        return jsonify({"status": "error", "msg": "数据集不存在"})
    return jsonify({"status": "success", "columns": dm.get_columns_by_type()})


# ==============================================================================
#  数据预处理路由
# ==============================================================================

@stats_bp.route('/api/stats/preprocess', methods=['POST'])
def stats_preprocess():
    """执行数据预处理操作"""
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id', '')
        operation = data.get('operation', '')
        params = data.get('params', {})

        dm = get_dataset(dataset_id)
        if not dm:
            return jsonify({"status": "error", "msg": "数据集不存在"})

        if operation == 'encode':
            result = Preprocessor.encode_variable(dm, params.get('col'), params.get('mapping'))
        elif operation == 'labels':
            result = Preprocessor.set_labels(dm, params.get('label_dict', {}))
        elif operation == 'detect_outliers':
            result = Preprocessor.detect_outliers(dm, params.get('cols', []), params.get('method', 'iqr'))
            return jsonify({"status": "success", "result": result, "info": dm.get_info()})
        elif operation == 'handle_outliers':
            result = Preprocessor.handle_outliers(dm, params.get('cols', []),
                                                   params.get('method', 'iqr'), params.get('action', 'delete'))
        elif operation == 'handle_missing':
            result = Preprocessor.handle_missing(dm, params.get('cols', []), params.get('method', 'delete'))
        elif operation == 'filter_samples':
            result = Preprocessor.filter_samples(dm, params.get('col'), params.get('condition'), params.get('value'))
        elif operation == 'generate_variable':
            result = Preprocessor.generate_variable(dm, params.get('source_cols', []),
                                                     params.get('operation', 'sum'), params.get('new_name', '新变量'))
        elif operation == 'standardize':
            result = Preprocessor.standardize(dm, params.get('cols', []), params.get('method', 'zscore'))
        else:
            return jsonify({"status": "error", "msg": f"未知操作: {operation}"})

        if isinstance(result, dict) and result.get('error'):
            return jsonify({"status": "error", "msg": result.get('msg', '操作失败')})

        return jsonify({"status": "success", "result": result, "info": dm.get_info()})
    except Exception as e:
        return jsonify({"status": "error", "msg": f"预处理失败: {str(e)}"})


@stats_bp.route('/api/stats/preprocess/undo', methods=['POST'])
def stats_preprocess_undo():
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id', '')
        dm = get_dataset(dataset_id)
        if not dm:
            return jsonify({"status": "error", "msg": "数据集不存在"})
        if dm.undo():
            return jsonify({"status": "success", "msg": "已撤销", "info": dm.get_info()})
        return jsonify({"status": "error", "msg": "无可撤销的操作"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


# ==============================================================================
#  统计分析路由
# ==============================================================================

@stats_bp.route('/api/stats/analyze', methods=['POST'])
def stats_analyze():
    """执行统计分析"""
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id', '')
        module = data.get('module', '')
        method = data.get('method', '')
        variables = data.get('variables', {})
        options = data.get('options', {})

        dm = get_dataset(dataset_id)
        if not dm:
            return jsonify({"status": "error", "msg": "数据集不存在"})
        df = dm.df

        result = None

        # 模块 2：描述性分析
        if module == 'descriptive':
            if method == 'frequency':
                col = variables.get('col') or (variables.get('cols', [None])[0])
                if not col:
                    return jsonify({"status": "error", "msg": "请选择分析变量"})
                result = DescriptiveAnalyzer.frequency(df, col)
            elif method == 'descriptive_stats':
                cols = variables.get('cols', [])
                group_var = variables.get('group_var')
                if not cols:
                    return jsonify({"status": "error", "msg": "请选择分析变量"})
                result = DescriptiveAnalyzer.descriptive_stats(df, cols, group_var)
            elif method == 'cross_tabulation':
                row_var = variables.get('row_var')
                col_var = variables.get('col_var')
                if not row_var or not col_var:
                    return jsonify({"status": "error", "msg": "请选择行变量和列变量"})
                result = DescriptiveAnalyzer.cross_tabulation(df, row_var, col_var)

        # 模块 3：差异比较
        elif module == 'comparison':
            if method == 'independent_ttest':
                value_col = variables.get('value_col')
                group_col = variables.get('group_col')
                if not value_col or not group_col:
                    return jsonify({"status": "error", "msg": "请选择连续变量和分组变量"})
                result = ComparisonAnalyzer.independent_ttest(df, value_col, group_col)
            elif method == 'paired_ttest':
                col1 = variables.get('col1')
                col2 = variables.get('col2')
                if not col1 or not col2:
                    return jsonify({"status": "error", "msg": "请选择两个配对变量"})
                result = ComparisonAnalyzer.paired_ttest(df, col1, col2)
            elif method == 'one_way_anova':
                value_col = variables.get('value_col')
                group_col = variables.get('group_col')
                if not value_col or not group_col:
                    return jsonify({"status": "error", "msg": "请选择连续变量和分组变量"})
                post_hoc = options.get('post_hoc', '')
                post_hoc = post_hoc if post_hoc else None
                result = ComparisonAnalyzer.one_way_anova(df, value_col, group_col, post_hoc)
            elif method == 'mann_whitney':
                value_col = variables.get('value_col')
                group_col = variables.get('group_col')
                if not value_col or not group_col:
                    return jsonify({"status": "error", "msg": "请选择连续变量和分组变量"})
                result = ComparisonAnalyzer.mann_whitney(df, value_col, group_col)
            elif method == 'wilcoxon_test':
                col1 = variables.get('col1')
                col2 = variables.get('col2')
                if not col1 or not col2:
                    return jsonify({"status": "error", "msg": "请选择两个配对变量"})
                result = ComparisonAnalyzer.wilcoxon_test(df, col1, col2)

        # 模块 4：相关与回归
        elif module == 'regression':
            if method == 'pearson_correlation':
                cols = variables.get('cols', [])
                if len(cols) < 2:
                    return jsonify({"status": "error", "msg": "请至少选择两个数值变量"})
                result = RegressionAnalyzer.pearson_correlation(df, cols)
            elif method == 'spearman_correlation':
                cols = variables.get('cols', [])
                if len(cols) < 2:
                    return jsonify({"status": "error", "msg": "请至少选择两个变量"})
                result = RegressionAnalyzer.spearman_correlation(df, cols)
            elif method == 'linear_regression':
                y_col = variables.get('y_col')
                x_cols = variables.get('x_cols', [])
                if not y_col or not x_cols:
                    return jsonify({"status": "error", "msg": "请选择因变量和自变量"})
                result = RegressionAnalyzer.linear_regression(df, y_col, x_cols)
            elif method == 'logistic_regression':
                y_col = variables.get('y_col')
                x_cols = variables.get('x_cols', [])
                if not y_col or not x_cols:
                    return jsonify({"status": "error", "msg": "请选择因变量和自变量"})
                max_iter = int(options.get('max_iter', 100))
                result = RegressionAnalyzer.logistic_regression(df, y_col, x_cols, max_iter)

        # 模块 5：前提检验
        elif module == 'prerequisite':
            if method == 'normality':
                cols = variables.get('cols', [])
                if not cols:
                    return jsonify({"status": "error", "msg": "请选择检验变量"})
                result = PrerequisiteTests.normality_test(df, cols)
            elif method == 'homogeneity':
                value_col = variables.get('value_col')
                group_col = variables.get('group_col')
                if not value_col or not group_col:
                    return jsonify({"status": "error", "msg": "请选择检验变量和分组变量"})
                result = PrerequisiteTests.homogeneity_test(df, value_col, group_col)

        # 模块 6：可视化
        elif module == 'chart':
            chart_title = options.get('title')
            if method == 'histogram':
                col = variables.get('col') or (variables.get('cols', [None])[0])
                bins = options.get('bins')
                result = {'charts': [ChartGenerator.histogram(df, col, bins, chart_title)]}
            elif method == 'kde':
                col = variables.get('col') or (variables.get('cols', [None])[0])
                result = {'charts': [ChartGenerator.kde_plot(df, col, chart_title)]}
            elif method == 'boxplot':
                cols = variables.get('cols', [])
                group_var = variables.get('group_var')
                result = {'charts': [ChartGenerator.boxplot(df, cols, group_var, chart_title)]}
            elif method == 'bar':
                col = variables.get('col') or (variables.get('cols', [None])[0])
                horizontal = options.get('horizontal', False)
                result = {'charts': [ChartGenerator.bar_chart(df, col, chart_title, horizontal)]}
            elif method == 'scatter':
                x_col = variables.get('x_col')
                y_col = variables.get('y_col')
                if not x_col or not y_col:
                    return jsonify({"status": "error", "msg": "请选择 X 和 Y 变量"})
                result = {'charts': [ChartGenerator.scatter_plot(df, x_col, y_col, chart_title)]}

        if result is None:
            return jsonify({"status": "error", "msg": f"未知的分析方法: {module}/{method}"})

        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "msg": f"分析失败: {str(e)}"})


# ==============================================================================
#  导出路由
# ==============================================================================

@stats_bp.route('/api/stats/export/table', methods=['POST'])
def stats_export_table():
    """导出统计结果表格"""
    try:
        data = request.get_json()
        fmt = data.get('format', 'csv')
        result = data.get('result', {})

        if fmt == 'csv':
            content = ExportManager.to_csv_bytes(result)
            return send_file(io.BytesIO(content), mimetype='text/csv',
                             as_attachment=True, download_name='统计结果.csv')
        elif fmt == 'excel':
            content = ExportManager.to_excel_bytes(result)
            return send_file(io.BytesIO(content),
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             as_attachment=True, download_name='统计结果.xlsx')
        return jsonify({"status": "error", "msg": "不支持的格式"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


@stats_bp.route('/api/stats/records_list')
def stats_records_list():
    """获取可导入的病历记录列表"""
    try:
        conn = _get_db()
        c = conn.cursor()
        c.execute('SELECT id, case_number, role_id, template_id, original_filename, create_time '
                  'FROM medical_records ORDER BY create_time DESC')
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return jsonify({"status": "success", "records": rows})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})
