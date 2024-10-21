import React, { useState, useEffect } from 'react';
import './navigation.css';
import avatar from '../../img/avatar.png';
import { useGlobalContext } from '../../Context/globalContext';
import { menuitems } from '../../Utils/menuitems';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function Navigation({ active, setActive }) {
    const [name, setName] = useState('');
    const URL = 'http://localhost:3001/api/v1/';
    const navigate = useNavigate();

    const getName = async () => {
        try {
            const userId = localStorage.getItem('userId');
            const response = await axios.get(`${URL}getName`, {
                headers: {
                    'user-id': userId
                }
            });
            setName(response.data.name);
        } catch (error) {
            console.error('Error fetching name:', error);
        }
    };

    useEffect(() => {
        getName();
    }, []);

    const handleSignOut = () => {
        localStorage.removeItem('userId');
        navigate('/login');
    };

    return (
        <div className='navigation-div'>
            <div className="user-con">
                <img src={avatar} alt="logo" />
                <div className="text">
                    <h2>{name}</h2>
                    <p>Your Money</p>
                </div>
            </div>
            <ul className='menu-items'>
                {menuitems.map((item) => {
                    return (
                        <li key={item.id} 
                            onClick={() => setActive(item.id)}
                            className={active === item.id ? 'active' : ''}
                        >
                            {item.icon}
                            <span>{item.title}</span>
                        </li>
                    );
                })}
            </ul>
            <div className="bottom-nav">
                <li>
                    <a href='/' onClick={handleSignOut}>
                        <i className="fa-solid fa-right-from-bracket"></i> Sign Out
                    </a>
                </li>
            </div>
        </div>
    );
}

export default Navigation;
