import React, { useState, useEffect } from 'react'
import './register.css'
import video from '../../Imgs/Rain.mp4'
import logo from '../../Imgs/logo4.png'
import { Link } from 'react-router-dom'
import { FaUserShield } from "react-icons/fa";
import { MdOutlineMailOutline } from "react-icons/md";
import { AiOutlineSwapRight } from "react-icons/ai";
import { BsFillShieldLockFill } from "react-icons/bs";
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Register = () => {
    const navigate=useNavigate()
    const [user, setUser] = useState([])
    const [email, setEmail] = useState('')
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')

    // useEffect(() => {
    //     fetchUsers();
    // }, [])

    // const fetchUsers = () => {   
    //     axios.get('http://localhost:3001/register')
    //         .then(result => console.log(result))
    //         .catch(err => console.log(err))
    // }

    const handleRegister = (event) => {
        event.preventDefault()
        axios.post('http://localhost:3001/register', {email, username, password})
        .then(() => {
            alert('Registration Successful')
            setEmail('')
            setUsername('')
            setPassword('')
            // fetchUsers()
            navigate('/')
        })  
        .catch((error) => {
            alert(error.response.data.error)
        })
    }
    return (
        <div className='register-page flex'>
            <div className='container flex'>

                <div className='videoDiv'>
                    <video src={video} autoPlay muted loop></video>
                    <div className='textDiv'>
                        <h2 className='title'>SmartSavings</h2>
                        <p>Beware of little expenses</p>
                    </div>
                    <div className='footerDiv flex'>
                        <span className='text'>Have an account?</span>
                        <Link to={'/'}>
                            <button className='btn'>Login</button>
                        </Link>
                    </div>
                </div>

                <div className="formDiv flex">
                    <div className="headerDiv">
                        <img src={logo} alt="Logo Img" />
                        <h3>Let Us Know You!</h3>
                    </div>
                    <form onSubmit={handleRegister} className='form grid'>

                        <div className="inputDiv">
                            <label htmlFor="email">Email</label>
                            <div className="input flex">
                                <MdOutlineMailOutline className='icon' />
                                <input
                                    type="email"
                                    id='Email'
                                    placeholder='Enter Email'
                                    onChange={(e) => setEmail(e.target.value)
                                    }
                                    required
                                />
                            </div>
                        </div>

                        <div className="inputDiv">
                            <label htmlFor="Username">Username</label>
                            <div className="input flex">
                                <FaUserShield className='icon' />
                                <input
                                    type="text"
                                    id='userName'
                                    placeholder='Enter Username'
                                    onChange={(e) => setUsername(e.target.value)}
                                    required
                                />
                            </div>
                        </div>


                        <div className="inputDiv">
                            <label htmlFor="password">Password</label>
                            <div className="input flex">
                                <BsFillShieldLockFill className='icon' />
                                <input
                                    type="password"
                                    id='password'
                                    placeholder='Enter Password'
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                />
                            </div>
                        </div>

                        <button type='submit' className='btn flex'>
                            <span>Register</span>
                            <AiOutlineSwapRight className='icon' />
                        </button>


                    </form>
                </div>
            </div>
        </div>
    )
}

export default Register