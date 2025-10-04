import pandas as pd

class MyStatistic:
    def find_orders_within_range(df, minValue, maxValue):
        # tổng giá trị từng đơn hàng
        df['line_total'] = df['UnitPrice'] * df['Quantity'] * (1 - df['Discount'])
        order_totals = df.groupby('OrderID')['line_total'].sum()
        # lọc đơn hàng trong range
        orders_within_range = order_totals[(order_totals >= minValue) & (order_totals <= maxValue)]
        # danh sách các mã đơn hàng không trùng nhau
        unique_orders = df[df['OrderID'].isin(orders_within_range.index)]['OrderID'].drop_duplicates().tolist()
        return unique_orders

    df=pd.read_csv('../dataset/SalesTransactions/SalesTransactions.csv')

    minValue=float(input('Nhập giá trị min:'))
    maxValue=float(input('Nhập giá trị max:'))
    result=find_orders_within_range(df, minValue, maxValue)
    print('Danh sách các hóa đơn trong phạm vi giá trị từ',minValue, 'đến', maxValue, 'là:', result)
