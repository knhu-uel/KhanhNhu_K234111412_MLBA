import sqlite3
import pandas as pd

# Helper chạy query và trả về DataFrame có header
def run_query(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=cols)

# (1) TOP N Invoice có tổng trị giá từ a -> b, sắp xếp giảm dần theo tổng
def top_invoices_in_range(conn, a, b, n):
    sql = """
        SELECT
            i.InvoiceId,
            i.CustomerId,
            i.InvoiceDate,
            i.BillingCountry,
            ROUND(i.Total, 2) AS Total
        FROM Invoice i
        WHERE i.Total BETWEEN ? AND ?
        ORDER BY i.Total DESC, i.InvoiceId ASC
        LIMIT ?
    """
    return run_query(conn, sql, (a, b, n))

# (2) TOP N khách hàng có nhiều Invoice nhất
def top_customers_by_invoice_count(conn, n):
    sql = """
        SELECT
            c.CustomerId,
            c.FirstName || ' ' || c.LastName AS CustomerName,
            COUNT(i.InvoiceId) AS InvoiceCount
        FROM Customer c
        LEFT JOIN Invoice i ON i.CustomerId = c.CustomerId
        GROUP BY c.CustomerId, CustomerName
        ORDER BY InvoiceCount DESC, CustomerName ASC
        LIMIT ?
    """
    return run_query(conn, sql, (n,))

# (3) TOP N khách hàng có tổng giá trị Invoice cao nhất
def top_customers_by_total_spend(conn, n):
    sql = """
        SELECT
            c.CustomerId,
            c.FirstName || ' ' || c.LastName AS CustomerName,
            ROUND(COALESCE(SUM(i.Total), 0), 2) AS TotalSpend,
            COUNT(i.InvoiceId) AS InvoiceCount
        FROM Customer c
        LEFT JOIN Invoice i ON i.CustomerId = c.CustomerId
        GROUP BY c.CustomerId, CustomerName
        ORDER BY TotalSpend DESC, CustomerName ASC
        LIMIT ?
    """
    return run_query(conn, sql, (n,))

# --- ví dụ chạy thử ---
if __name__ == "__main__":
    conn = sqlite3.connect("databases/Chinook_Sqlite.sqlite")

    # (1) top hóa đơn trong khoảng 5 -> 20$
    print("\n#1 TOP Invoices in range [5, 20] (DESC):")
    print(top_invoices_in_range(conn, 5, 20, 10).to_string(index=False))

    # (2) top khách hàng theo số lượng invoice
    print("\n#2 TOP Customers by Invoice Count:")
    print(top_customers_by_invoice_count(conn, 10).to_string(index=False))

    # (3) top khách hàng theo tổng chi tiêu
    print("\n#3 TOP Customers by Total Spend:")
    print(top_customers_by_total_spend(conn, 10).to_string(index=False))

    conn.close()
