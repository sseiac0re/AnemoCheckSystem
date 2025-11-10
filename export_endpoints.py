from flask import make_response, redirect, url_for, flash
from flask_login import current_user
import csv
from io import StringIO

import database as db
from timezone_utils import format_philippines_time_ampm

def register_export_routes(app):
    @app.route('/export/history.csv')
    def export_history_csv():
        # Require login at view layer if flask-login is used; here we check current_user
        try:
            if not getattr(current_user, 'is_authenticated', False):
                return redirect(url_for('login'))
        except Exception:
            # If current_user not available, allow route to proceed (likely in tests)
            pass

        records = db.get_user_classification_history(current_user.id, limit=10000) if getattr(current_user, 'is_authenticated', False) else []

        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(['Date', 'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'NEUTROPHILS', 'LYMPHOCYTES', 'MONOCYTES', 'EOSINOPHILS', 'BASOPHIL', 'IMMATURE_GRANULOCYTES', 'Classification', 'Confidence', 'Notes'])
        for r in records:
            confidence_value = r.get('confidence')
            confidence_formatted = f"{float(confidence_value)*100:.2f}%" if confidence_value is not None else ''
            created_at_formatted = '\t' + format_philippines_time_ampm(r.get('created_at')) if r.get('created_at') else ''
            # Use 0.8 (median from training data) as default if value is None/empty
            # Explicitly check for None vs 0 - 0 should be exported as 0, None should default to 0.8
            immature_granulocytes_raw = r.get('immature_granulocytes')
            # Handle None, empty string, or actual 0 value
            # Check if value is None, empty string, or not provided
            if immature_granulocytes_raw is None or immature_granulocytes_raw == '' or (isinstance(immature_granulocytes_raw, str) and immature_granulocytes_raw.strip() == ''):
                immature_granulocytes = 0.8  # Default when value is None/empty in database
            else:
                # Convert to float - this handles both int 0 and float 0.0
                try:
                    immature_granulocytes = float(immature_granulocytes_raw)
                    # Explicitly preserve 0.0 values (don't convert to default)
                    # This ensures that explicit 0 values entered by user are preserved
                except (ValueError, TypeError):
                    # If conversion fails, use default
                    immature_granulocytes = 0.8
            cw.writerow([
                created_at_formatted, r.get('wbc'), r.get('rbc'), r.get('hgb'), r.get('hct'), r.get('mcv'), r.get('mch'), r.get('mchc'), r.get('plt'),
                r.get('neutrophils'), r.get('lymphocytes'), r.get('monocytes'), r.get('eosinophils'), r.get('basophil'), immature_granulocytes, r.get('predicted_class'), confidence_formatted, r.get('notes')
            ])

        output = make_response(si.getvalue())
        output.headers['Content-Disposition'] = f'attachment; filename=anemocheck_history_{getattr(current_user, "username", "user")}.csv'
        output.headers['Content-type'] = 'text/csv'
        return output

    @app.route('/admin/export/users.csv')
    def admin_export_users_csv():
        if not getattr(current_user, 'is_authenticated', False) or not getattr(current_user, 'is_admin', False):
            flash('Access denied.')
            return redirect(url_for('admin_dashboard'))

        users = db.get_all_users(limit=10000)
        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(['id','username','email','first_name','last_name','gender','date_of_birth','medical_id','is_admin','created_at','last_login'])
        for u in users:
            created_at_formatted = '\t' + format_philippines_time_ampm(u.get('created_at')) if u.get('created_at') else ''
            last_login_formatted = '\t' + format_philippines_time_ampm(u.get('last_login')) if u.get('last_login') else ''
            cw.writerow([u.get('id'), u.get('username'), u.get('email'), u.get('first_name'), u.get('last_name'), u.get('gender'), u.get('date_of_birth'), u.get('medical_id'), u.get('is_admin'), created_at_formatted, last_login_formatted])

        output = make_response(si.getvalue())
        output.headers['Content-Disposition'] = 'attachment; filename=anemocheck_users.csv'
        output.headers['Content-type'] = 'text/csv'
        return output

    @app.route('/admin/export/classification_history.csv')
    def admin_export_classification_history_csv():
        if not getattr(current_user, 'is_authenticated', False) or not getattr(current_user, 'is_admin', False):
            flash('Access denied.')
            return redirect(url_for('admin_dashboard'))

        records = db.get_all_classification_history(limit=100000)
        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(['id','user_id','username','created_at','wbc','rbc','hgb','hct','mcv','mch','mchc','plt','neutrophils','lymphocytes','monocytes','eosinophils','basophil','immature_granulocytes','predicted_class','confidence','recommendation','notes'])
        for r in records:
            confidence_value = r.get('confidence')
            confidence_formatted = f"{float(confidence_value)*100:.2f}%" if confidence_value is not None else ''
            created_at_formatted = '\t' + format_philippines_time_ampm(r.get('created_at')) if r.get('created_at') else ''
            # Use 0.8 (median from training data) as default if value is None/empty
            # Explicitly check for None vs 0 - 0 should be exported as 0, None should default to 0.8
            immature_granulocytes_raw = r.get('immature_granulocytes')
            # Handle None, empty string, or actual 0 value
            # Check if value is None, empty string, or not provided
            if immature_granulocytes_raw is None or immature_granulocytes_raw == '' or (isinstance(immature_granulocytes_raw, str) and immature_granulocytes_raw.strip() == ''):
                immature_granulocytes = 0.8  # Default when value is None/empty in database
            else:
                # Convert to float - this handles both int 0 and float 0.0
                try:
                    immature_granulocytes = float(immature_granulocytes_raw)
                    # Explicitly preserve 0.0 values (don't convert to default)
                    # This ensures that explicit 0 values entered by user are preserved
                except (ValueError, TypeError):
                    # If conversion fails, use default
                    immature_granulocytes = 0.8
            cw.writerow([r.get('id'), r.get('user_id'), r.get('username'), created_at_formatted, r.get('wbc'), r.get('rbc'), r.get('hgb'), r.get('hct'), r.get('mcv'), r.get('mch'), r.get('mchc'), r.get('plt'), r.get('neutrophils'), r.get('lymphocytes'), r.get('monocytes'), r.get('eosinophils'), r.get('basophil'), immature_granulocytes, r.get('predicted_class'), confidence_formatted, r.get('recommendation'), r.get('notes')])

        output = make_response(si.getvalue())
        output.headers['Content-Disposition'] = 'attachment; filename=anemocheck_classification_history.csv'
        output.headers['Content-type'] = 'text/csv'
        return output

    @app.route('/admin/export/medical_data.csv')
    def admin_export_medical_data_csv():
        if not getattr(current_user, 'is_authenticated', False) or not getattr(current_user, 'is_admin', False):
            flash('Access denied.')
            return redirect(url_for('admin_dashboard'))

        conn = db.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT md.user_id, u.username, md.height, md.weight, md.blood_type, md.medical_conditions, md.medications, md.updated_at
            FROM medical_data md
            LEFT JOIN users u ON md.user_id = u.id
        """)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(['user_id','username','height','weight','blood_type','medical_conditions','medications','updated_at'])
        for r in rows:
            updated_at_formatted = '\t' + format_philippines_time_ampm(r.get('updated_at')) if r.get('updated_at') else ''
            cw.writerow([r.get('user_id'), r.get('username'), r.get('height'), r.get('weight'), r.get('blood_type'), r.get('medical_conditions'), r.get('medications'), updated_at_formatted])

        output = make_response(si.getvalue())
        output.headers['Content-Disposition'] = 'attachment; filename=anemocheck_medical_data.csv'
        output.headers['Content-type'] = 'text/csv'
        return output

# helper to register when imported
def init_app(app):
    register_export_routes(app)
