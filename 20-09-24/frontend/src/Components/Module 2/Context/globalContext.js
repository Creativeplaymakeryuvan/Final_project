import React, { useContext, useState } from "react"
import axios from "axios"
import Loading from "../Components/Loading/Loading";
import { CategoryScale } from "chart.js";

const URL = "http://localhost:3001/api/v1/";

const GlobalContext = React.createContext()

export const GlobalProvider = ({ children }) => {

    const [incomes, setIncomes] = useState([])
    const [expenses, setExpenses] = useState([])
    const [error, setError] = useState(null)
    const [loading, setLoading] = useState(false);


    // for Income

    const addIncome = async (income) => {
        const response = await axios.post(`${URL}add-income`, income)
            .then(() => {
                alert('Income added')
            })
            .catch((err) => {
                alert('Income not added')
                setError(err.response.data.message)
            })
        getIncomes()
    }

    const getIncomes = async () => {
        const userId = localStorage.getItem("userId")
        try {
            const response = await axios.get(`${URL}get-incomes/${userId}`)

            setIncomes(response.data)
            console.log("Icome data" + response.data)
        } catch (err) {
            setError(err.response.data.message)
        }
    }

    const deleteIncome = async (userId, incomeId, updatedFields) => {
        const res = await axios.delete(`${URL}delete-income/${userId}/${incomeId}`, updatedFields)
        getIncomes()
    }

    const updateIncome = async (userId, incomeId, updatedFields) => {
        try {
            const res = await axios.put(`${URL}update-income/${userId}/${incomeId}`, updatedFields)
        } catch (err) {
            setError(err.response.data.message)
        }
        getIncomes()
    }

    const totalIncome = () => {
        let totalIncome = 0;
        incomes.forEach((income) => {
            totalIncome += income.amount
        })
        return totalIncome;
    }

    // for Expense

    const addExpense = async (expense) => {
        console.log('expese:='+expense)
        const response = await axios.post(`${URL}add-expense`, expense)
            .then(() => {
                alert('Expense added')
            })
            .catch((err) => {
                alert('Expense not added')
                setError(err.response.data.message)
            })
        getExpenses()
    }

    const getExpenses = async () => {
        const userId = localStorage.getItem("userId")
        try {
            const response = await axios.get(`${URL}get-expenses/${userId}`)

            setExpenses(response.data)
            console.log("Expense data" + response.data)
        } catch (err) {
            setError(err.response.data.message)
        }
    }

    const deleteExpense = async (userId, expenseId) => {
        const res = await axios.delete(`${URL}delete-expense/${userId}/${expenseId}`)
        getExpenses()
    }

    const updateExpense = async (userId, expenseId, updatedFields) => {
        try {
            const res = await axios.put(`${URL}update-expense/${userId}/${expenseId}`, updatedFields)
        } catch (err) {
            setError(err.response.data.message)
        }
        getExpenses()
    }

    const totalIncomes = () => {
        let totalIncome = 0;
        incomes.forEach((income) => {
            totalIncome += income.amount
        })
        return totalIncome;
    }

    const totalExpenses = () => {
        let totalExpense = 0;
        expenses.forEach((expense) => {
            totalExpense += expense.amount
        })
        return totalExpense;
    }

    const totalBalance = () => {
        return totalIncome() - totalExpenses()
    }

    const getCumulativeSavingsBeforeMonth = (month, year) => {
        let cumulativeSavings = 0;
    
        const startYear = 1990;
    
        for (let y = startYear; y <= year; y++) {
            const finalMonth = (y === year) ? month : 12;
    
            for (let m = 1; m <= finalMonth; m++) {
                const { totalMonthlyIncome, totalMonthlyExpenses } = getMonthlyTransactions(m, y);
    
                cumulativeSavings += totalMonthlyIncome - totalMonthlyExpenses;
            }
        }
        return cumulativeSavings;
    };
    


    const getMonthlySavings = () => {
        const savingsPerMonth = [];
    
        const formatMonthYear = (date) => {
            const d = new Date(date);
            const month = d.getMonth() + 1;
            const year = d.getFullYear();
            return { month, year };
        };
    
        const findMonthYear = (month, year) => {
            return savingsPerMonth.find(saving => saving.month === month && saving.year === year);
        };
    
        incomes.forEach((income) => {
            const { month, year } = formatMonthYear(income.date);
            let monthlyRecord = findMonthYear(month, year);
    
            if (monthlyRecord) {
                monthlyRecord.totalIncome += income.amount;
            } else {
                savingsPerMonth.push({
                    month,
                    year,
                    totalIncome: income.amount,
                    totalExpenses: 0
                });
            }
        });
    
        expenses.forEach((expense) => {
            const { month, year } = formatMonthYear(expense.date);
            let monthlyRecord = findMonthYear(month, year);
    
            if (monthlyRecord) {
                monthlyRecord.totalExpenses += expense.amount;
            } else {
                savingsPerMonth.push({
                    month,
                    year,
                    totalIncome: 0,
                    totalExpenses: expense.amount
                });
            }
        });
    
        savingsPerMonth.forEach((record) => {
            record.savings = record.totalIncome - record.totalExpenses;
        });
    
        return savingsPerMonth;
    };
    
    
    const processBillImage = async (file) => {
        setLoading(true);
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await axios.post(`${URL}process-bill-image`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            const { inference } = response.data;
            console.log(inference)
            const amount = inference.prediction.totalAmount?.value || 'Amount not available';
            const date = inference.prediction.date?.value || 'Date not available';
            const totalItemsPurchased = inference.prediction.lineItems?.length || 0;
            const tax = inference.prediction.totalTax?.value || 0;
            const description = inference.prediction.lineItems?.map((item, index) => ({
                item: item.description || `Item ${index + 1}`,
                quantity: item.quantity|| 'N/A',
                price: item.totalAmount || 'N/A',
            })) || [];
            const title = inference.prediction.supplierName?.value || 'Store name not available';

            return {
                amount,
                date,
                totalItemsPurchased,
                description,
                title,
                category: "other",
                tax
            };

            // console.log('Total Amount:', amount);
            // console.log('Purchase Date:', purchaseDate);
            // console.log('Total Items Purchased:', totalItemsPurchased);
            // console.log('Items List:', itemsList);
            // console.log('Store Name:', storeName);

        } catch (error) {
            console.error('Error processing bill image:', error);
        } finally {
            setLoading(false);
        }
    };

    const processAndExtractReceiptData = async (file) => {
        setLoading(true);
        const formData = new FormData();
        formData.append('image', file);
    
        try {
            const billImageResponse = await axios.post(`${URL}process-bill-image`, formData);
    
            const { inference } = billImageResponse.data;
    
            const description = inference.prediction.lineItems?.map((item, index) => ({
                item: item.description || `Item ${index + 1}`,
                quantity: item.quantity || 'N/A',
                price: item.totalAmount || 'N/A',
            })) || [];
            const totalItemsPurchased = inference.prediction.lineItems?.length || 0;
            const tax = inference.prediction.totalTax?.value || 0;
            const date = inference.prediction.date?.value || 'Date not available';
    
            // Send a request to your Flask server to extract additional data
            const extractDataResponse = await axios.post('http://localhost:5000/extract-data', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
    
            const { title, totalprice } = extractDataResponse.data.extracted_data;
    
            return {
                title: title || 'Store name not available',
                date: date || 'Date not available',
                amount: totalprice || 'Amount not available',
                description,
                totalItemsPurchased,
                category: "other",
                tax,
            };
        } catch (error) {
            console.error('Error processing or extracting receipt data:', error.response ? error.response.data : error.message);
            return null;
        } finally {
            setLoading(false);
        }
    };

    const uploadImage = async (file) => {
        setLoading(true);
        setError(null);
        const formData = new FormData();
        formData.append('image', file); // Append the file to FormData

        try {
            const response = await axios.post('http://localhost:5000/extract-data', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            return response.data; // Return the response data
        } catch (err) {
            setError(err.response ? err.response.data : err.message);
            console.error('Image upload error:', error);
        } finally {
            setLoading(false);
        }
    };

    
    
    
    
    
    


    const processInvoice = async (file) => {
        const formData = new FormData();
        formData.append('file', file);
    
        try {
            const response = await axios.post(`${URL}process-invoice`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
    
            const { inference } = response.data;
    
            const amount = inference.prediction.totalAmount?.value || 'Amount not available';
            const date = inference.prediction.date?.value || 'Date not available';
            const totalItemsPurchased = inference.prediction.lineItems?.length || 0;
            const description = inference.prediction.lineItems?.map((item, index) => ({
                item: item.description?.value || `Item ${index + 1}`,
                quantity: item.quantity?.value || 'N/A',
                price: item.totalAmount?.value || 'N/A',
            })) || [];
            const title = inference.prediction.supplierName?.value || 'Supplier name not available';
    
            return {
                amount,
                date,
                totalItemsPurchased,
                description,
                title,
            };
    
            // Uncomment to debug
            // console.log('Total Amount:', amount);
            // console.log('Date:', date);
            // console.log('Total Items Purchased:', totalItemsPurchased);
            // console.log('Description:', description);
            // console.log('Supplier Name:', title);
    
        } catch (error) {
            console.error('Error processing invoice:', error);
        }
    };
    

    const transactionHistory = () => {
        const history = [...incomes, ...expenses]
        history.sort((a, b) => {
            return new Date(b.date) - new Date(a.date)
        })
        return history
    }


    const getMonthlyTransactions = (month, year) => {
        const filteredIncomes = incomes.filter((income) => {
            const incomeDate = new Date(income.date);
            return (
                incomeDate.getMonth() === month - 1 && incomeDate.getFullYear() === year
            );
        });

        const filteredExpenses = expenses.filter((expense) => {
            const expenseDate = new Date(expense.date);
            return (
                expenseDate.getMonth() === month - 1 && expenseDate.getFullYear() === year
            );
        });

        const totalMonthlyIncome = filteredIncomes.reduce(
            (acc, income) => acc + income.amount,
            0
        );

        const totalMonthlyExpenses = filteredExpenses.reduce(
            (acc, expense) => acc + expense.amount,
            0
        );

        const filteredtransactionHistory = () => {
            const history = [...filteredIncomes, ...filteredExpenses];
            history.sort((a, b) => {
                const dateA = new Date(a.date);
                const dateB = new Date(b.date);
                return dateB - dateA;
            });
            return history;
        }


        const totalTransactions = filteredIncomes.length + filteredExpenses.length;

        

        return {
            filteredIncomes,
            filteredExpenses,
            totalMonthlyIncome,
            totalMonthlyExpenses,
            totalTransactions,
            balance: totalMonthlyIncome - totalMonthlyExpenses,
            filteredtransactionHistory: filteredtransactionHistory()
        };
    };

    return (
        <GlobalContext.Provider value={{
            addIncome,
            getIncomes,
            incomes,
            deleteIncome,
            updateIncome,
            totalIncome,
            addExpense,
            getExpenses,
            expenses,
            deleteExpense,
            updateExpense,
            totalExpenses,
            totalIncomes,
            totalBalance,
            transactionHistory,
            getMonthlyTransactions,
            processBillImage,
            processAndExtractReceiptData,
            getCumulativeSavingsBeforeMonth,
            getMonthlySavings,
            uploadImage
        }}>
            {loading && <Loading />}
            {children}
        </GlobalContext.Provider>
    )
}

export const useGlobalContext = () => {
    return useContext(GlobalContext)
}