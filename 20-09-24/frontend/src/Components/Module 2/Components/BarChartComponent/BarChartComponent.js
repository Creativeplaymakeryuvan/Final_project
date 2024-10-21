import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useGlobalContext } from '../../Context/globalContext';

const BarChartComponent = ({ incomeData, expenseData, flag }) => {
    const { incomes, expenses } = useGlobalContext();
    if (!flag) {
        incomeData = incomes;
        expenseData = expenses;
    }

    const formatDate = (dateString) => {
        const date = new Date(dateString);
        const options = { day: '2-digit'};
        return date.toLocaleDateString('en-US', options).replace(',', ''); 
    };

    const chartData = [
        ...incomeData.map((income) => ({
            date: income.date,
            formattedDate: formatDate(income.date),
            income: income.amount,
            expense: 0,
        })),
        ...expenseData.map((expense) => ({
            date: expense.date,
            formattedDate: formatDate(expense.date),
            income: 0,
            expense: expense.amount,
        })),
    ];

    const aggregatedData = chartData.reduce((acc, curr) => {
        const existing = acc.find((item) => item.date === curr.date);
        if (existing) {
            existing.income += curr.income;
            existing.expense += curr.expense;
        } else {
            acc.push(curr);
        }
        return acc;
    }, []);

    const sortedData = aggregatedData.sort((a, b) => new Date(a.date) - new Date(b.date));

    const CustomTick = ({ x, y, payload }) => (
        <text x={x} y={y} textAnchor="middle" fill="#666" dy={10}>
            {payload.value}
        </text>
    );

    const isDataEmpty = incomeData.length === 0 && expenseData.length === 0;

    const placeholderData = [
        { index: 1, expense:0 },
        { index: 2 },
        { index: 3 },
        { index: 4 },
        { index: 5 },
        { index: 6 },
        { index: 7 },
        { index: 8 },
        { index: 9 },
        { index: 10, expense:0}
    ];

    return (
        <ResponsiveContainer width="100%" height={400}>
            <BarChart
                data={isDataEmpty ? placeholderData : sortedData}
                margin={{
                    top: 20, right: 30, left: 20, bottom: 50,
                }}
            >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                    dataKey={isDataEmpty ? "index" : "formattedDate"} 
                    tick={<CustomTick />}
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="income" fill="#82ca9d" />
                <Bar dataKey="expense" fill="#8884d8" />
            </BarChart>
        </ResponsiveContainer>
    );
};

export default BarChartComponent;
