import React, { useState } from 'react';
import './transaction.css';
import Button from '../Button/Button';
import { useGlobalContext } from '../../Context/globalContext';
import { FaTrashCan } from "react-icons/fa6";
import { FaPen } from "react-icons/fa";
import { dateFormet } from '../../Utils/dateFormet';
import UpdateModal from '../Models/Modal';
import { FaExclamationTriangle } from "react-icons/fa";

const ITEMS_PER_PAGE = 10;

function Transaction() {
    const { transactionHistory, deleteIncome, deleteExpense, updateIncome, updateExpense } = useGlobalContext();
    const userId = localStorage.getItem('userId');

    const [showModal, setShowModal] = useState(false);
    const [currentData, setCurrentData] = useState(null);
    const [currentPage, setCurrentPage] = useState(0);
    const [sortType, setSortType] = useState('both'); // 'income', 'expense', 'both'
    const [sortDate, setSortDate] = useState('none'); // 'recent', 'oldest', 'none'
    const [sortAmount, setSortAmount] = useState('none'); // 'high', 'low', 'none'

    const handleClose = () => setShowModal(false);

    const handleShow = (data) => {
        setCurrentData(data); 
        setShowModal(true);
    };

    const handleSubmit = (updatedData) => {
        if (currentData.type === 'income') {
            updateIncome(userId, currentData._id, updatedData);
        } else {
            updateExpense(userId, currentData._id, updatedData);
        }
        handleClose();
    };

    const transactions = transactionHistory();

    let sortedTransactions = transactions.filter((trans) => {
        if (sortType === 'both') return true;
        return trans.type === sortType;
    });

    if (sortDate !== 'none') {
        sortedTransactions = sortedTransactions.sort((a, b) => {
            if (sortDate === 'recent') {
                return new Date(b.date) - new Date(a.date); 
            }
            return new Date(a.date) - new Date(b.date); 
        });
    }

    if (sortAmount !== 'none') {
        sortedTransactions = sortedTransactions.sort((a, b) => {
            if (sortAmount === 'high') {
                return b.amount - a.amount;
            }
            return a.amount - b.amount;
        });
    }

    const totalPages = Math.ceil(sortedTransactions.length / ITEMS_PER_PAGE);

    const currentTransactions = sortedTransactions.slice(
        currentPage * ITEMS_PER_PAGE,
        (currentPage + 1) * ITEMS_PER_PAGE
    );

    const handleNextPage = () => {
        if (currentPage < totalPages - 1) {
            setCurrentPage(prevPage => prevPage + 1);
        }
    };

    const handlePrevPage = () => {
        if (currentPage > 0) {
            setCurrentPage(prevPage => prevPage - 1);
        }
    };

    return (
        <div className='InnerLayout'>
            <div className='header-container'>
                <h2>All Transactions</h2>
                {/* Sorting */}
                <div className='sorting-controls'>
                    <div>
                        <label>Sort by Type:</label>
                        <select value={sortType} onChange={(e) => setSortType(e.target.value)}>
                            <option value="both">Both</option>
                            <option value="income">Income</option>
                            <option value="expense">Expense</option>
                        </select>
                    </div>
                    <div>
                        <label>Sort by Date:</label>
                        <select value={sortDate} onChange={(e) => setSortDate(e.target.value)}>
                            <option value="recent">Recent First</option>
                            <option value="oldest">Oldest First</option>
                        </select>
                    </div>
                    <div>
                        <label>Sort by Amount:</label>
                        <select value={sortAmount} onChange={(e) => setSortAmount(e.target.value)}>
                            <option value="none">None</option>
                            <option value="high">High to Low</option>
                            <option value="low">Low to High</option>
                        </select>
                    </div>
                </div>
            </div>

            {transactions.length === 0 ? (
                <div className="no-data">
                    <FaExclamationTriangle className="no-data-icon" />
                    <p className='no-transactions-message'>No data available. Please add some records.</p>
                </div>
            ) : (
                <>
                    <div className='Table-head'>
                        <h2>Title</h2>
                        <h2>Amount</h2>
                        <h2>Type</h2>
                        <h2>Category</h2>
                        <h2>Date</h2>
                        <h2>Action</h2>
                    </div>
                    {currentTransactions.map((data) => (
                        <div className='Table-body' key={data._id}>
                            <h2>{data.title}</h2>
                            <h2>{data.amount}</h2>
                            <h2>{data.type}</h2>
                            <h2>{data.category.split(" - ")[1] ? data.category.split(" - ")[1] : data.category.split(" - ")[0]}</h2>
                            <h2>{dateFormet(data.date)}</h2>
                            <h2 className='action-con'>
                                <Button
                                    className=''
                                    icon={<FaPen />}
                                    bg={"transparent"}
                                    onClick={() => handleShow(data)}
                                />
                                <Button
                                    className=''
                                    icon={<FaTrashCan className='Table-logo' />}
                                    bg={"transparent"}
                                    onClick={() => data.type === 'income' ? deleteIncome(userId, data._id) : deleteExpense(userId, data._id)}
                                />
                            </h2>
                        </div>
                    ))}

                    <div className="pagination">
                        <button onClick={handlePrevPage} disabled={currentPage === 0}>
                            Previous
                        </button>
                        <span>Page {currentPage + 1} of {totalPages}</span>
                        <button onClick={handleNextPage} disabled={currentPage >= totalPages - 1}>
                            Next
                        </button>
                    </div>
                </>
            )}

            {showModal && currentData && (
                <UpdateModal
                    show={showModal}
                    handleClose={handleClose}
                    currentData={currentData}
                    onSubmit={handleSubmit}
                />
            )}
        </div>
    );
}

export default Transaction;
