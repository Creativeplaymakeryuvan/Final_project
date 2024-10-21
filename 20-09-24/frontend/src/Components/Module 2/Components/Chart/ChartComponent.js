import React, { useRef } from 'react'
import {
    Chart as ChartJs,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { useGlobalContext } from '../../Context/globalContext'
import { dateFormet } from '../../Utils/dateFormet'
import './chart.css'

ChartJs.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
)

function ChartComponent() {
    const { incomes, expenses } = useGlobalContext();
    const chartRef = useRef(null);

    const data = {
        labels: incomes.map((inc) => dateFormet(inc.date)),
        datasets: [
            {
                label: 'Income',
                data: incomes.map((income) => income.amount),
                backgroundColor: 'green',
                tension: 0.2,
            },
            {
                label: 'Expenses',
                data: expenses.map((expense) => expense.amount),
                backgroundColor: 'red',
                tension: 0.2,
            },
        ],
    };

    return (
        <div className="chart-div">
            <Line ref={chartRef} data={data} />
        </div>
    );
}

export default ChartComponent
