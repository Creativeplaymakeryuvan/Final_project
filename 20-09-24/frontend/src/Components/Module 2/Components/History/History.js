import React from 'react';
import { useGlobalContext } from '../../Context/globalContext';
import './history.css';
import { FaSadTear } from 'react-icons/fa';  // Import an icon from react-icons

function History({ filteredtransactionHistory, flag }) {
  const { transactionHistory } = useGlobalContext();
  const history = flag ? filteredtransactionHistory : transactionHistory();

  return (
    <div>
      <h2>Recent History</h2>
      <div className='history-div'>
        {history.length === 0 ? (
          <div className="no-history">
            <FaSadTear size={50} color="gray" /> 
            <p>Oops! No transaction found</p>
          </div>
        ) : (
          history.map((item) => {
            const { _id, title, amount, type } = item;
            return (
              <div key={_id} className="history-item">
                <p
                  style={{
                    color: type === 'expense' ? 'red' : 'green',
                  }}
                >
                  {title}
                </p>
                <p
                  style={{
                    color: type === 'income' ? 'green' : 'red',
                  }}
                >
                  {type === 'expense' ? `-${amount}` : `+${amount}`}
                </p>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default History;
