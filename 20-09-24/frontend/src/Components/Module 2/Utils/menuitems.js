import { GrMoney } from "react-icons/gr";
export const menuitems = [
    {
        id: 1,
        title: 'Dashboard',
        icon: <i class="fa-solid fa-house"></i>,
        link: '/dashboard'
    },
    // {
    //     id: 2,
    //     title: 'Budgets',
    //     icon: <i><GrMoney /></i>,
    //     link: '/dashboard'
    // },
    {
        id: 2,
        title: 'View Transactions',
        icon: <i className="fa-solid fa-credit-card"></i>,
        link: '/dashboard'
    },
    {
        id: 3,
        title: 'Incomes',
        icon: <i className="fa-solid fa-money-bill-trend-up"></i>,
        link: '/dashboard'
    },
    {
        id: 4,
        title: 'Expenses',
        icon: <i className="fa-solid fa-money-bill-transfer"></i>,
        link: '/dashboard'
    }
]