import pandas as pd

class MyStatistic:
    def find_orders_within_range(df, minValue, maxValue, sortType=True):
        df['line_total'] = df['UnitPrice'] * df['Quantity'] * (1 - df['Discount'])
        order_totals = df.groupby('OrderID')['line_total'].sum().reset_index(name="Sum")
        filtered = order_totals[(order_totals["Sum"] >= minValue) & (order_totals["Sum"] <= maxValue)]
        filtered = filtered.sort_values(by="Sum", ascending=sortType).reset_index(drop=True)
        return filtered

if __name__ == "__main__":
    df = pd.read_csv('../dataset/SalesTransactions/SalesTransactions.csv')

    minValue = float(input('Nhập giá trị min: '))
    maxValue = float(input('Nhập giá trị max: '))
    sortType = input("Sort tăng (True) hay giảm (False)? ").strip().lower() == "true"

    result = MyStatistic.find_orders_within_range(df, minValue, maxValue, sortType)

    print(f"\nDanh sách các hóa đơn trong phạm vi giá trị từ {minValue} đến {maxValue}:")
    print(result.to_string(index=False))
